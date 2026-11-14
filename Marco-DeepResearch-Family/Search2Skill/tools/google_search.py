"""
Google Search tool via direct HTTP request to search proxy API.

Supports batch query input (list of strings) with parallel execution.

Usage as standalone:
    python tools/google_search.py "what is machine learning"
    python tools/google_search.py "query1" "query2" "query3"

Usage in code:
    from tools.google_search import GoogleSearchTool
    tool = GoogleSearchTool()
    result = tool.call(["what is machine learning"])
    # result: {"results": {query: [...]}, "formatted": str}
    # Backward compatible:
    results = tool.search("what is machine learning")
    # results: [{"title": ..., "url": ..., "content": ..., "score": ...}, ...]
"""

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import requests

PROXY_API_URL = os.getenv("SEARCH_PROXY_API_URL", "")
DEFAULT_AUTHORIZATION = os.getenv("SEARCH_PROXY_AUTHORIZATION", "")
GOOGLE_SEARCH_CACHE_DIR = os.getenv(
    "EXSKILL_GOOGLE_SEARCH_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "google_search_cache"),
)

logger = logging.getLogger(__name__)

_cache_io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gcache")
_CACHE_IO_TIMEOUT = 10


# ==================== Sharded File Cache ====================
# Files distributed across 256 subdirs (first 2 hex chars of MD5)
# All NAS I/O in thread pool to prevent asyncio event loop blocking.

def _sharded_cache_path(base_dir: str, key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(base_dir, h[:2], h + ".json")


def _load_cached_search(cache_key: str) -> Optional[List[Dict]]:
    def _read():
        path = _sharded_cache_path(GOOGLE_SEARCH_CACHE_DIR, cache_key)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    try:
        data = _cache_io_pool.submit(_read).result(timeout=_CACHE_IO_TIMEOUT)
        if data is not None:
            logger.info(f"[GoogleSearch] cache hit: {cache_key[:80]}")
        return data
    except Exception:
        return None


def _store_cached_search(cache_key: str, query: str, results: List[Dict]) -> None:
    def _write():
        try:
            path = _sharded_cache_path(GOOGLE_SEARCH_CACHE_DIR, cache_key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(results, f, ensure_ascii=False)
            os.rename(tmp, path)
        except Exception as exc:
            logger.warning(f"[GoogleSearch] cache write failed: {exc}")
    _cache_io_pool.submit(_write)


# ==================== GoogleSearchTool ====================

class GoogleSearchTool:
    """Google search via proxy API with batch query support.

    Primary interface: ``call(querys)`` accepts a list of query strings
    and returns structured results with per-query grouping.

    Backward compatible: ``search(query)`` still works for single-query calls.
    """

    def __init__(
        self,
        authorization: str = DEFAULT_AUTHORIZATION,
        job_name: str = "default",
        backend: str = "proxy",
        enable_cache: bool = True,
        max_results: int = 5,
        max_content_length: int = 3000,
        timeout: int = 30,
        max_retries: int = 3,
        max_workers: int = 3,
    ):
        self.authorization = authorization
        self.job_name = job_name
        self.backend = backend
        self.enable_cache = enable_cache
        self.max_results = max_results
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers

    # ==================== Tool Registration Interface ====================

    @property
    def description(self) -> str:
        return "Performs web searches using Google. Supports single or multiple query strings and returns top search results."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "querys": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of query strings for batch searching."
                }
            },
            "required": ["querys"],
        }

    # ==================== Primary Interface ====================

    def call(self, querys: Union[str, List[str]]) -> Dict[str, Any]:
        """Unified entry point: execute Google search (supports batch).

        Args:
            querys: A single query string or a list of query strings.

        Returns:
            {
                "results": {query_str: [{"title": ..., "link": ..., "snippet": ..., "position": ...}, ...]},
                "formatted": str,   # human-readable formatted results
            }
        """
        if isinstance(querys, str):
            querys = [querys]

        logger.info(f"[GoogleSearch] Batch search {len(querys)} queries, max {self.max_results} per query")

        # 1. Batch cache lookup
        cached_results: Dict[str, List[Dict]] = {}
        uncached_querys: List[str] = []

        if self.enable_cache:
            for q in querys:
                cache_key = self._cache_key(q)
                cached = _load_cached_search(cache_key)
                if cached is not None:
                    cached_results[q] = cached
                else:
                    uncached_querys.append(q)
        else:
            uncached_querys = list(querys)

        # 2. Parallel search for uncached queries
        search_results: Dict[str, List[Dict]] = {}
        failed_queries: List[str] = []

        if uncached_querys:
            search_results, failed_queries = self._batch_search(uncached_querys)

            # Write new results to cache
            if self.enable_cache:
                for q in uncached_querys:
                    if q not in failed_queries and q in search_results:
                        cache_key = self._cache_key(q)
                        _store_cached_search(cache_key, q, search_results[q])

        # 3. Merge cached + new results
        all_results: Dict[str, List[Dict]] = {}
        all_results.update(cached_results)
        all_results.update(search_results)

        # 4. Format
        total_count = sum(len(v) for v in all_results.values() if isinstance(v, list))
        formatted = self._format_results(all_results, total_count, querys, failed_queries)

        return {
            "results": all_results,
            "formatted": formatted,
        }

    # ==================== Backward Compatible Interface ====================

    def search(self, query: str) -> List[Dict]:
        """Search a single query, returns unified format [{title, url, content, score}].

        Backward compatible wrapper around ``call()``.
        """
        result = self.call([query])
        items = result["results"].get(query, [])
        # Convert internal format {title, link, snippet, position} to legacy format {title, url, content, score}
        return [self._to_legacy_format(item) for item in items]

    # ==================== Batch Search ====================

    def _batch_search(self, querys: List[str]) -> tuple:
        """Execute multiple queries in parallel via ThreadPoolExecutor.

        Returns:
            (results_dict, failed_queries) where results_dict maps query -> list of result dicts.
        """
        all_results: Dict[str, List[Dict]] = {}
        failed_queries: List[str] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._search_single, querys))

        for query, parsed_results, error in results:
            if error:
                failed_queries.append(query)
                all_results[query] = []
            else:
                all_results[query] = parsed_results

        total = sum(len(v) for v in all_results.values())
        logger.info(
            f"[GoogleSearch] Batch search done: {total} results, "
            f"{len(failed_queries)} failed out of {len(querys)} queries"
        )

        return all_results, failed_queries

    def _search_single(self, query: str) -> tuple:
        """Search a single query via proxy API.

        Returns:
            (query, parsed_results, error) where error is None on success.
        """
        headers = {
            "Authorization": self.authorization,
            "Content-Type": "application/json",
        }
        payload = {"q": query + " -filetype:pdf"}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    PROXY_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                parsed = self._parse_response(response.text)
                return (query, parsed, None)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[GoogleSearch] Attempt {attempt + 1}/{self.max_retries} failed for '{query}': {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        logger.error(f"[GoogleSearch] All retries failed for: {query}, last error: {last_error}")
        return (query, [], "[GoogleSearchTool] invoke failed.")

    # ==================== Parsing ====================

    def _parse_response(self, raw_text: str) -> List[Dict]:
        """Parse proxy API response into standardized format.

        Returns JSON with ``organic`` field (standard format).
        Each item: {title, link, snippet, position}
        """
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return self._parse_formatted_text(raw_text)

        results = []

        items = data.get("organic") or data.get("items") or data.get("results") or []

        if isinstance(items, list) and items:
            for i, item in enumerate(items[:self.max_results]):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", "") or item.get("url", ""),
                    "snippet": (
                        item.get("snippet", "")
                        or item.get("description", "")
                        or item.get("content", "")
                    )[:self.max_content_length],
                    "position": item.get("position", i + 1),
                })
        elif isinstance(data, dict) and "result" in data:
            results = self._parse_formatted_text(str(data["result"]))

        return results

    def _parse_formatted_text(self, text: str) -> List[Dict]:
        """Parse formatted text into standardized structure (fallback parser).

        Format:
            1. Title|URL
            Snippet text
        """
        results = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and len(line) > 2 and line[0].isdigit() and ". " in line:
                dot_pos = line.index(". ")
                entry = line[dot_pos + 2:]
                if "|" in entry:
                    title, url = entry.rsplit("|", 1)
                else:
                    title, url = entry, ""

                snippet = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not (next_line[0].isdigit() and ". " in next_line) and not next_line.startswith("Query:"):
                        snippet = next_line
                        i += 1

                results.append({
                    "title": title.strip(),
                    "link": url.strip(),
                    "snippet": snippet[:self.max_content_length],
                    "position": len(results) + 1,
                })

                if len(results) >= self.max_results:
                    break
            i += 1

        return results

    # ==================== Formatting ====================

    def _format_results(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        total_count: int,
        querys: List[str],
        failed_queries: List[str],
    ) -> str:
        """Format search results into human-readable string (token-optimized)."""
        lines = []
        lines.append(f"[Search] Queries:{len(querys)} Results:{total_count} Failed:{len(failed_queries)}")

        for query in querys:
            items = results.get(query, [])
            lines.append("")
            lines.append(f"Query: {query} ({len(items)} results)")

            for i, item in enumerate(items, 1):
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                lines.append(f"{i}. {title}|{link}")
                if snippet:
                    lines.append(snippet)

        if failed_queries:
            lines.append("Failed: " + ", ".join(failed_queries))

        return "\n".join(lines)

    # ==================== Helpers ====================

    def _cache_key(self, query: str) -> str:
        return json.dumps(
            {
                "query": query,
                "backend": self.backend,
                "max_results": self.max_results,
                "max_content_length": self.max_content_length,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    @staticmethod
    def _to_legacy_format(item: Dict) -> Dict:
        """Convert internal format {title, link, snippet, position} to legacy {title, url, content, score}."""
        return {
            "title": item.get("title", ""),
            "url": item.get("link", "") or item.get("url", ""),
            "content": item.get("snippet", "") or item.get("content", ""),
            "score": 0.0,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tools/google_search.py <query1> [query2] ...")
        sys.exit(1)

    querys = sys.argv[1:]
    tool = GoogleSearchTool()

    result = tool.call(querys)
    print(result["formatted"])
