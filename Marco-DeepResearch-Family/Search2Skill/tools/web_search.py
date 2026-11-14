"""
Web search tool supporting multiple search engines.

Supports: tavily / duckduckgo / google (search proxy API).
Returns unified format: [{title, url, content, score}].

Batch search: ``batch_search(querys)`` accepts a list of query strings.
"""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse


ACADEMIC_DOMAINS = (
    "arxiv.org", "aclanthology.org", "openreview.net", "nature.com",
    "science.org", "ieee.org", "acm.org", ".edu", ".gov",
)


# Domains observed to consistently return 403/Forbidden from the page fetcher.
# Filtered at search-time so they never reach the model, saving context and
# avoiding wasted PAGE_VISIT calls. Extend as new offenders are identified.
ALWAYS_FORBIDDEN_DOMAINS = (
    "mathoverflow.net",
    "math.stackexchange.com",
    "stats.stackexchange.com",
    "physics.stackexchange.com",
    "cs.stackexchange.com",
    "stackoverflow.com",
    "stackexchange.com",
    "www.sciencedirect.com",
    "www.scribd.com",
    "www.researchgate.net",
    "askfilo.com",
    "www.youtube.com",
)


class WebSearchTool:
    """Web search returning unified result format."""

    def __init__(self, engine: str = "tavily", tavily_api_key: str = "",
                 max_results: int = 5, max_content_length: int = 3000,
                 search_depth: str = "advanced",
                 google_authorization: str = "",
                 google_job_name: str = "",
                 google_backend: str = "proxy",
                 google_enable_cache: bool = True,
                 blocked_domains: Optional[List[str]] = None):
        self.engine = engine
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.search_depth = search_depth
        # Merge user-supplied blocked_domains with hardcoded ALWAYS_FORBIDDEN_DOMAINS
        # (sites like mathoverflow / *.stackexchange that consistently 403 the fetcher).
        _user_blocked = [d.lower() for d in (blocked_domains or [])]
        _always_blocked = [d.lower() for d in ALWAYS_FORBIDDEN_DOMAINS]
        self.blocked_domains = list(dict.fromkeys(_user_blocked + _always_blocked))

        if self.engine == "tavily":
            from tavily import TavilyClient
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        elif self.engine == "google":
            from tools.google_search import GoogleSearchTool
            self.google_client = GoogleSearchTool(
                authorization=google_authorization or "",
                job_name=google_job_name or "default",
                backend=google_backend,
                enable_cache=google_enable_cache,
                max_results=max_results,
                max_content_length=max_content_length,
            )

    @classmethod
    def from_config(cls, config) -> "WebSearchTool":
        ws = config["web_search"]
        return cls(
            engine=ws["engine"],
            tavily_api_key=ws.get("tavily_api_key", ""),
            max_results=ws["max_results"],
            max_content_length=ws["max_content_length"],
            search_depth=ws.get("search_depth", "advanced"),
            google_authorization=ws.get("google_authorization", ""),
            google_job_name=ws.get("google_job_name", ""),
            google_backend=ws.get("google_backend", "proxy"),
            google_enable_cache=ws.get("google_enable_cache", True),
            blocked_domains=ws.get("blocked_domains", []),
        )

    def _is_blocked(self, url: str) -> bool:
        """Check if a URL belongs to a blocked domain."""
        if not self.blocked_domains or not url:
            return False
        domain = urlparse(url).netloc.lower()
        return any(domain == bd or domain.endswith("." + bd) for bd in self.blocked_domains)

    async def search(self, query: str) -> List[Dict]:
        """Execute search, return unified format [{title, url, content, score}]."""
        if self.engine == "tavily":
            results = self._search_tavily(query)
        elif self.engine == "google":
            results = self._search_google(query)
        else:
            results = self._search_duckduckgo(query)
        # Filter out blocked domains
        if self.blocked_domains:
            before = len(results)
            results = [r for r in results if not self._is_blocked(r.get("url", ""))]
            filtered = before - len(results)
            if filtered:
                print(f"[WebSearch] Filtered {filtered} results from blocked domains")
        return results

    async def batch_search(self, querys: Union[str, List[str]]) -> Dict[str, List[Dict]]:
        """Execute batch search for multiple queries, return per-query results.

        Args:
            querys: A single query string or a list of query strings.

        Returns:
            Dict mapping each query string to its unified-format result list.
        """
        if isinstance(querys, str):
            querys = [querys]

        if self.engine == "google":
            return self._batch_search_google(querys)

        # Fallback: sequential search for non-google engines
        all_results: Dict[str, List[Dict]] = {}
        for q in querys:
            all_results[q] = await self.search(q)
        return all_results

    def _search_tavily(self, query: str) -> List[Dict]:
        response = self.tavily_client.search(
            query=query,
            max_results=self.max_results,
            search_depth=self.search_depth,
            include_answer=True,
            include_raw_content=False,
        )
        results = []
        if response.get("answer"):
            results.append({
                "title": "[Tavily AI Summary]",
                "url": "",
                "content": response["answer"],
                "score": 1.0,
            })
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")[:self.max_content_length],
                "score": item.get("score", 0.0),
            })
        return results

    def _search_google(self, query: str) -> List[Dict]:
        return self.google_client.search(query)

    def _batch_search_google(self, querys: List[str]) -> Dict[str, List[Dict]]:
        """Batch search via GoogleSearchTool.call(), returns per-query legacy-format results."""
        result = self.google_client.call(querys)
        per_query = result.get("results", {})
        # Convert internal format to legacy format + filter blocked domains
        output: Dict[str, List[Dict]] = {}
        for q in querys:
            items = per_query.get(q, [])
            legacy = [self.google_client._to_legacy_format(item) for item in items]
            if self.blocked_domains:
                legacy = [r for r in legacy if not self._is_blocked(r.get("url", ""))]
            output[q] = legacy
        return output

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=self.max_results))
        return [{
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "content": r.get("body", "")[:self.max_content_length],
            "score": 0.0,
        } for r in raw_results]


# ==================== URL Selection Helpers ====================

def dedup_results(items: List[Dict], top_k: int) -> List[Dict]:
    """Deduplicate search results by URL."""
    out, seen = [], set()
    for it in items:
        url = (it.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(it)
        if len(out) >= top_k:
            break
    return out


def _tokenize(text: str) -> set:
    stop = {
        "the", "a", "an", "is", "are", "to", "of", "and", "or",
        "in", "on", "for", "with", "what", "how", "why", "when",
        "where", "which",
    }
    tokens = set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))
    return {t for t in tokens if len(t) > 1 and t not in stop}


def heuristic_select(query: str, goal: str, candidates: List[Dict], fetch_top_k: int) -> List[int]:
    """Select URLs by keyword overlap + domain bonus + position bonus."""
    target_tokens = _tokenize(query + " " + goal)
    ranked: List[Tuple[float, int]] = []

    for idx, item in enumerate(candidates):
        title = item.get("title", "")
        snippet = item.get("content", "")
        url = item.get("url", "")
        domain = (urlparse(url).netloc or "").lower()

        text_tokens = _tokenize(f"{title} {snippet}")
        overlap = len(target_tokens.intersection(text_tokens))

        domain_bonus = 3.0 if any(d in domain for d in ACADEMIC_DOMAINS) else 0.0
        position_bonus = max(0.0, 1.5 - 0.2 * idx)

        score = overlap * 1.0 + domain_bonus + position_bonus
        ranked.append((score, idx))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in ranked[:fetch_top_k]]


async def llm_select_urls(
    llm_client,
    query: str,
    goal: str,
    candidates: List[Dict],
    max_pick: int,
) -> List[int]:
    """Use LLM to select the most relevant URLs from search candidates.
    Returns 0-based indexes of selected candidates.
    """
    lines = []
    for idx, item in enumerate(candidates, start=1):
        lines.append(
            f"[{idx}] title={item.get('title', '')}\n"
            f"url={item.get('url', '')}\n"
            f"snippet={item.get('content', '')[:500]}"
        )

    candidates_block = "\n\n".join(lines)
    prompt = dedent(
        f"""\
        You rank search results for research browsing.
        Select the most useful pages to read for the target goal.

        Query: {query}
        Goal: {goal}
        Max pick: {max_pick}

        Candidates:
        {candidates_block}

        Output strict JSON only:
        {{"selected_ids":[1,2],"reason":"short reason"}}

        Rules:
        - IDs must be unique.
        - IDs must be between 1 and candidate count.
        - Number of selected IDs must be at most max pick.
        """
    ).strip()

    try:
        response = await llm_client.chat([{"role": "user", "content": prompt}])
        if not response:
            return []

        parsed = _safe_json_parse(response)
        ids = parsed.get("selected_ids", []) if isinstance(parsed, dict) else []

        valid_ids = []
        n = len(candidates)
        for i in ids:
            if isinstance(i, int) and 1 <= i <= n and i not in valid_ids:
                valid_ids.append(i)
            if len(valid_ids) >= max_pick:
                break

        return [i - 1 for i in valid_ids]
    except Exception as e:
        print(f"[LLMUrlSelector] fallback to heuristic: {e}")
        return []


def _safe_json_parse(text: str) -> Dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    left = text.find("{")
    right = text.rfind("}")
    if left != -1 and right != -1 and right > left:
        try:
            return json.loads(text[left: right + 1])
        except Exception:
            return {}
    return {}
