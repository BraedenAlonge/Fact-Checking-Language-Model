"""Search-provider adapters and URL normalization."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
}


class SearchError(RuntimeError):
    """Raised when a configured search provider cannot return results."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> list[SearchResult]: ...


def canonicalize_url(value: str) -> str:
    """Remove fragments and common tracking fields for stable deduplication."""

    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return ""
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return ""
    hostname = parsed.hostname.casefold().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = hostname
    if port and not (
        (parsed.scheme.casefold() == "http" and port == 80)
        or (parsed.scheme.casefold() == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    parameters = []
    for key, value_item in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered.startswith("utm_") or lowered in TRACKING_PARAMETERS:
            continue
        parameters.append((key, value_item))
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (parsed.scheme.casefold(), netloc, path, urlencode(parameters), "")
    )


def deduplicate_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    selected: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        normalized = canonicalize_url(result.url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(
            SearchResult(
                title=" ".join(result.title.split()),
                url=normalized,
                snippet=" ".join(result.snippet.split()),
                provider=result.provider,
            )
        )
        if len(selected) >= limit:
            break
    return selected


class BraveSearchProvider:
    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, *, timeout: float = 15.0) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        self.timeout = timeout
        if not self.api_key:
            raise SearchError(
                "BRAVE_SEARCH_API_KEY is required when --provider brave is selected"
            )

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        limit = max(1, min(limit, 20))
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query[:600],
            "count": limit,
            "country": "US",
            "search_lang": "en",
            "safesearch": "moderate",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False, trust_env=False
            ) as client:
                response = await client.get(self.endpoint, headers=headers, params=params)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"Brave Search request failed: {exc}") from exc
        rows = payload.get("web", {}).get("results", [])
        results = [
            SearchResult(
                title=str(row.get("title") or "Untitled result"),
                url=str(row.get("url") or ""),
                snippet=str(row.get("description") or ""),
                provider=self.name,
            )
            for row in rows
        ]
        return deduplicate_results(results, limit)


class DDGSSearchProvider:
    name = "ddgs"

    def __init__(
        self,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        backend: str = "auto",
        timeout: int = 10,
    ) -> None:
        self.region = region
        self.safesearch = safesearch
        self.backend = backend
        self.timeout = timeout

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise SearchError(
                "DDGS is not installed; run the prototype requirements install "
                "or select --provider brave"
            ) from exc

        def run_search() -> list[dict[str, Any]]:
            return DDGS(timeout=self.timeout).text(
                query,
                region=self.region,
                safesearch=self.safesearch,
                max_results=limit,
                backend=self.backend,
            )

        try:
            rows = await asyncio.to_thread(run_search)
        except Exception as exc:
            raise SearchError(f"DDGS search failed: {exc}") from exc
        results = [
            SearchResult(
                title=str(row.get("title") or "Untitled result"),
                url=str(row.get("href") or row.get("url") or ""),
                snippet=str(row.get("body") or row.get("snippet") or ""),
                provider=self.name,
            )
            for row in rows
        ]
        return deduplicate_results(results, limit)


class StaticSearchProvider:
    """Deterministic provider used by tests and offline demonstrations."""

    name = "static"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    async def search(self, query: str, limit: int) -> list[SearchResult]:
        del query
        return deduplicate_results(self.results, limit)


def create_search_provider(name: str) -> SearchProvider:
    if name == "brave":
        return BraveSearchProvider()
    if name == "ddgs":
        return DDGSSearchProvider()
    raise SearchError(f"unknown search provider: {name}")
