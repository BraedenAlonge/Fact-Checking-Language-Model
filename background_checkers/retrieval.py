"""Bounded and defensive web-page retrieval for prototype evidence packets."""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urljoin, urlsplit

import httpx

from .search import SearchResult, canonicalize_url


WORD = re.compile(r"\b[\w'-]+\b")
BLOCKED_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer"}


class RetrievalError(RuntimeError):
    """Raised when a page is unsafe or cannot be converted to evidence text."""


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocked_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.casefold()
        if lowered in BLOCKED_TAGS:
            self.blocked_depth += 1
        if lowered == "title":
            self.in_title = True
        if lowered in {"p", "div", "article", "section", "li", "br", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in BLOCKED_TAGS and self.blocked_depth:
            self.blocked_depth -= 1
        if lowered == "title":
            self.in_title = False
        if lowered in {"p", "div", "article", "section", "li", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.blocked_depth:
            return
        if self.in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)

    @property
    def title(self) -> str:
        return normalize_space(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        blocks = [normalize_space(part) for part in "".join(self.text_parts).splitlines()]
        return "\n".join(block for block in blocks if block)


@dataclass
class RetrievedSource:
    source_id: str
    title: str
    url: str
    source_type: str
    excerpt: str
    domain: str
    query: str
    provider: str
    fetch_status: str

    def model_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
            "excerpt": self.excerpt,
        }

    def report_dict(self) -> dict[str, str]:
        return asdict(self)


def normalize_space(value: str) -> str:
    decoded = html.unescape(value)
    if any(marker in decoded for marker in ("Ã", "Â", "â€", "â€“", "â€™")):
        try:
            decoded = decoded.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"[ \t\r\f\v]+", " ", decoded).strip()


def source_type_for_url(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").casefold()
    if hostname.endswith(".gov") or ".gov." in hostname:
        return "government"
    if hostname.endswith(".edu") or ".edu." in hostname:
        return "education"
    if hostname.endswith("wikipedia.org"):
        return "wikipedia"
    if any(name in hostname for name in ("pubmed", "semanticscholar", "arxiv")):
        return "scientific"
    return "web"


def domain_for_url(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").casefold().rstrip(".")
    return hostname[4:] if hostname.startswith("www.") else hostname


async def ensure_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise RetrievalError("only absolute HTTP(S) URLs are allowed")
    hostname = parsed.hostname
    try:
        port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
    except ValueError as exc:
        raise RetrievalError("URL contains an invalid port") from exc
    try:
        literal = ipaddress.ip_address(hostname)
        addresses = [literal]
    except ValueError:
        try:
            rows = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise RetrievalError(f"DNS lookup failed for {hostname}") from exc
        addresses = list({ipaddress.ip_address(row[4][0]) for row in rows})
    if not addresses or any(not address.is_global for address in addresses):
        raise RetrievalError(f"URL resolves to a non-public address: {hostname}")


def _candidate_windows(text: str, width: int = 1200) -> Iterable[str]:
    paragraphs = [normalize_space(part) for part in text.splitlines() if part.strip()]
    if not paragraphs:
        paragraphs = [normalize_space(text)]
    for paragraph in paragraphs:
        if len(paragraph) <= width:
            yield paragraph
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + width)
            if end < len(paragraph):
                boundary = paragraph.rfind(" ", start, end)
                if boundary > start + width // 2:
                    end = boundary
            yield paragraph[start:end].strip()
            start = max(end, start + 1)


def select_excerpt(text: str, claim: str, query: str, limit: int = 1600) -> str:
    target = {token.casefold() for token in WORD.findall(f"{claim} {query}") if len(token) > 2}
    windows = [window for window in _candidate_windows(text) if window]
    if not windows:
        return ""

    def score(window: str) -> tuple[float, int]:
        words = {token.casefold() for token in WORD.findall(window)}
        overlap = len(words & target)
        density = overlap / max(len(words), 1)
        return overlap + density, -len(window)

    selected = max(windows, key=score)
    return selected[:limit].strip()


class PageRetriever:
    def __init__(
        self,
        *,
        timeout: float = 12.0,
        max_bytes: int = 2_000_000,
        concurrency: int = 4,
        max_redirects: int = 3,
        user_agent: str = "BackgroundCheckersPrototype/0.1 (course research)",
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.concurrency = concurrency
        self.max_redirects = max_redirects
        self.user_agent = user_agent

    async def _download(self, client: httpx.AsyncClient, initial_url: str) -> tuple[str, bytes, str]:
        url = canonicalize_url(initial_url)
        if not url:
            raise RetrievalError("invalid result URL")
        for _ in range(self.max_redirects + 1):
            await ensure_public_url(url)
            async with client.stream(
                "GET",
                url,
                headers={"User-Agent": self.user_agent, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1"},
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise RetrievalError("redirect did not provide a location")
                    url = canonicalize_url(urljoin(url, location))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").casefold()
                if not (content_type.startswith("text/html") or content_type.startswith("text/plain")):
                    raise RetrievalError(f"unsupported content type: {content_type or 'unknown'}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        raise RetrievalError("page exceeded retrieval byte limit")
                return url, bytes(body), content_type
        raise RetrievalError("page exceeded redirect limit")

    async def retrieve_one(
        self,
        result: SearchResult,
        source_id: str,
        claim: str,
        query: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        *,
        fetch_pages: bool,
    ) -> RetrievedSource:
        title = normalize_space(result.title) or "Untitled result"
        excerpt = normalize_space(result.snippet)
        url = canonicalize_url(result.url)
        status = "snippet_only"
        if fetch_pages:
            try:
                async with semaphore:
                    final_url, body, content_type = await self._download(client, url)
                url = final_url
                encoding = "utf-8"
                match = re.search(r"charset=([\w.-]+)", content_type)
                if match:
                    encoding = match.group(1)
                decoded = body.decode(encoding, errors="replace")
                if content_type.startswith("text/html"):
                    parser = VisibleTextParser()
                    parser.feed(decoded)
                    page_text = parser.text
                    if parser.title:
                        title = parser.title
                else:
                    page_text = normalize_space(decoded)
                selected = select_excerpt(page_text, claim, query)
                if selected:
                    excerpt = selected
                    status = "page_text"
                else:
                    status = "empty_page_fallback_to_snippet"
            except Exception as exc:
                status = f"snippet_fallback:{type(exc).__name__}"
        return RetrievedSource(
            source_id=source_id,
            title=title,
            url=url,
            source_type=source_type_for_url(url),
            excerpt=excerpt,
            domain=domain_for_url(url),
            query=query,
            provider=result.provider,
            fetch_status=status,
        )

    async def retrieve(
        self,
        results: list[SearchResult],
        claim: str,
        query: str,
        *,
        fetch_pages: bool = True,
        source_offset: int = 0,
    ) -> list[RetrievedSource]:
        semaphore = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            return await asyncio.gather(
                *(
                    self.retrieve_one(
                        result,
                        f"s{source_offset + index + 1}",
                        claim,
                        query,
                        client,
                        semaphore,
                        fetch_pages=fetch_pages,
                    )
                    for index, result in enumerate(results)
                )
            )
