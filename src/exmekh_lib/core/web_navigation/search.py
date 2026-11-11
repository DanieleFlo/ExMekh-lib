"""Deterministic public-web search used to discover page URLs."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import zip_longest
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from exmekh_lib.config.web import WEB_NAVIGATION
from exmekh_lib.config.web_text import WEB_TEXT

from . import image_discovery, ranking
from .errors import WebNavigationError

if TYPE_CHECKING:
    from PIL import Image

MAX_SEARCH_RESULTS = WEB_NAVIGATION.max_search_results
SEARCH_CANDIDATE_LIMIT = WEB_NAVIGATION.search_candidate_limit
SEARCH_BACKENDS = WEB_NAVIGATION.search_backends
_TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_LOGGER = logging.getLogger(__name__)


def search_web(
    query: str,
    *,
    max_results: int = WEB_NAVIGATION.default_search_results,
    timeout_seconds: int = WEB_NAVIGATION.default_timeout_seconds,
) -> dict[str, object]:
    """Search several non-Google providers and return bounded candidate pages."""

    normalized_query = _validate_search_arguments(
        query=query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )
    candidates = _retrieve_candidates(normalized_query, timeout_seconds=timeout_seconds)
    ranked = ranking.rank_results(normalized_query, candidates)
    return _build_search_payload(
        query=normalized_query,
        candidates=candidates,
        ranked=ranked,
        max_results=max_results,
    )


def search_web_multimodal(
    query: str,
    *,
    local_image: Image.Image,
    max_results: int = WEB_NAVIGATION.default_search_results,
    timeout_seconds: int = WEB_NAVIGATION.default_timeout_seconds,
) -> dict[str, object]:
    """Search with text discovery and an optional native visual reranking stage."""

    normalized_query = _validate_search_arguments(
        query=query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )
    candidates = _retrieve_candidates(normalized_query, timeout_seconds=timeout_seconds)
    preliminary = ranking.rank_results(normalized_query, candidates)
    selected_indices = [
        index for index, _ in preliminary[: image_discovery.IMAGE_CANDIDATE_LIMIT]
    ]
    discovered_images = image_discovery.discover_result_images(
        candidates,
        selected_indices,
        timeout_seconds=timeout_seconds,
    )
    image_urls: dict[int, str] = {}
    visual_ranking_applied = False
    try:
        image_ranking, image_urls = ranking.compute_multimodal_ranking(
            normalized_query,
            candidates,
            local_image=local_image,
            discovered_images=discovered_images,
        )
    except Exception as error:  # noqa: BLE001 - visual ranking is an optional signal.
        _LOGGER.warning(WEB_TEXT.native_visual_ranking_unavailable, error)
        ranked = preliminary
    else:
        visual_ranking_applied = True
        ranked = ranking.rank_results(
            normalized_query, candidates, image_ranking=image_ranking
        )
    return _build_search_payload(
        query=normalized_query,
        candidates=candidates,
        ranked=ranked,
        max_results=max_results,
        image_urls=image_urls,
        visual_ranking_applied=visual_ranking_applied,
    )


def _validate_search_arguments(
    *, query: str, max_results: int, timeout_seconds: int
) -> str:
    """Validate public search arguments and return the normalized query."""
    normalized_query = query.strip() if isinstance(query, str) else ""
    if not normalized_query:
        raise WebNavigationError(
            WEB_TEXT.search_query_must_be_a_non_empty_string, kind="invalid_query"
        )
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= MAX_SEARCH_RESULTS
    ):
        raise WebNavigationError(
            WEB_TEXT.max_results_must_be_between_and.format(
                max_search_results=MAX_SEARCH_RESULTS
            ),
            kind="invalid_max_results",
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= WEB_NAVIGATION.max_timeout_seconds
    ):
        raise WebNavigationError(
            WEB_TEXT.timeout_seconds_must_be_between_and.format(
                web_navigation_max_timeout_seconds=WEB_NAVIGATION.max_timeout_seconds
            ),
            kind="invalid_timeout",
        )
    return normalized_query


def _retrieve_candidates(
    normalized_query: str, *, timeout_seconds: int
) -> list[dict[str, object]]:
    """Collect each provider independently so one failure cannot discard other results."""
    backends = SEARCH_BACKENDS.split(",")

    def fetch(backend: str) -> list[dict[str, object]]:
        return DDGS(timeout=timeout_seconds).text(
            normalized_query,
            max_results=SEARCH_CANDIDATE_LIMIT,
            backend=backend,
        )

    provider_results: list[list[dict[str, object]]] = []
    last_failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=len(backends)) as executor:
        futures = [executor.submit(fetch, backend) for backend in backends]
        for future in futures:
            try:
                raw_results = future.result()
            except (DDGSException, OSError) as error:
                last_failure = error
                continue
            provider_results.append(
                [
                    normalized
                    for item in raw_results or []
                    if (normalized := _normalize_result(item)) is not None
                ]
            )

    if not provider_results:
        raise WebNavigationError(
            WEB_TEXT.public_web_search_failed_across_the_available_providers,
            kind="search_failed",
        ) from last_failure

    normalized_results = [
        item
        for row in zip_longest(*provider_results)
        for item in row
        if item is not None
    ]
    return _deduplicate_results(
        normalized_results,
        limit=SEARCH_CANDIDATE_LIMIT,
    )


def _build_search_payload(
    *,
    query: str,
    candidates: list[dict[str, object]],
    ranked: list[tuple[int, float]],
    max_results: int,
    image_urls: dict[int, str] | None = None,
    visual_ranking_applied: bool | None = None,
) -> dict[str, object]:
    """Build the stable public response from one final candidate ranking."""
    results: list[dict[str, object]] = []
    for index, score in ranked[:max_results]:
        candidate = dict(candidates[index])
        candidate["relevance_score"] = score
        if image_urls and index in image_urls:
            candidate["image_url"] = image_urls[index]
        results.append(candidate)
    payload: dict[str, object] = {
        "operation": "search",
        "query": query,
        "results": results,
        "result_count": len(results),
        "max_results": max_results,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "discovery_only": True,
        "untrusted_content": True,
    }
    if visual_ranking_applied is not None:
        payload["visual_ranking_applied"] = visual_ranking_applied
        payload["visual_images_used"] = (
            len(image_urls or {}) if visual_ranking_applied else 0
        )
    return payload


def _normalize_result(item: object) -> dict[str, object] | None:
    """Normalize one provider result and omit unusable destination URLs."""
    if not isinstance(item, dict):
        return None
    url = _canonicalize_url(str(item.get("href", "")).strip())
    if url is None:
        return None
    return {
        "title": str(item.get("title", "")).strip(),
        "url": url,
        "snippet": str(item.get("body", "")).strip(),
    }


def _canonicalize_url(url: str) -> str | None:
    """Normalize one result URL and remove tracking-only distinctions."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    netloc_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = netloc_host if port is None else f"{netloc_host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_PARAMETERS
            and not key.lower().startswith("utm_")
        ],
        doseq=True,
    )
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


def _deduplicate_results(
    candidates: list[dict[str, object]], *, limit: int
) -> list[dict[str, object]]:
    """Return ranked candidates after canonical URL and title/host deduplication."""
    results: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    for candidate in candidates:
        url = str(candidate["url"])
        parsed = urlparse(url)
        path = re.sub(r"/videos$", "", parsed.path.rstrip("/")) or "/"
        url_key = urlunparse(parsed._replace(path=path, query="", fragment=""))
        title_key = re.sub(r"\W+", " ", str(candidate["title"]).lower()).strip()
        host_title = (str(parsed.hostname or "").removeprefix("www."), title_key)
        if url_key in seen_urls or (title_key and host_title in seen_titles):
            continue
        seen_urls.add(url_key)
        seen_titles.add(host_title)
        results.append(candidate)
        if len(results) >= limit:
            break
    return results
