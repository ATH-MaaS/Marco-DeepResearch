"""
Web page visit tool: fetch page content and extract relevant information.

Supports multiple fetch modes:
- proxy: Search proxy service (default)
- requests: Direct HTTP request with HTML-to-Markdown conversion
- jina: Jina Reader API (interface kept for future open-source use)

Uses a compression LLM to extract goal-relevant information from raw page content.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import brotli
import requests as http_requests
from markdownify import MarkdownConverter, chomp

logger = logging.getLogger(__name__)

# ==================== Config ====================

PROXY_READ_URL = os.getenv(
    "SEARCH_PROXY_READ_URL", ""
)
JINA_READER_URL_PREFIX = os.getenv("JINA_READER_URL_PREFIX", "https://r.jina.ai/")
RAW_PAGE_CACHE_ENABLED = os.getenv("EXSKILL_RAW_PAGE_CACHE_ENABLED", "true").lower() == "true"
RAW_PAGE_CACHE_DIR = os.getenv(
    "EXSKILL_RAW_PAGE_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "raw_page_cache"),
)
_page_cache_io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pcache")
_PAGE_CACHE_IO_TIMEOUT = 10


def _truncate_content(content: str, max_length: int) -> str:
    """Truncate content keeping head and tail for cache storage."""
    if len(content) <= max_length:
        return content
    half = max_length // 2
    return (
        content[:half]
        + "\n..._This content has been truncated..._\n"
        + content[-half:]
    )

# ==================== Prompt ====================

WEBPAGE_EXTRACTOR_PROMPT = """Extract information from the webpage based on the goal.

## Content
{webpage_content}

## Goal
{goal}

## Instructions
1. Locate sections in the content that are relevant to the goal.
2. Extract key facts, data, and necessary context into "content":
   - For factual queries: provide specific answers with source details
   - For verification: state whether the claim is supported, contradicted, or not mentioned
   - For broad topics: summarize the most important points
3. If relevant information is not found, set found=false.
4. Extract supplementary domain knowledge into "supplementary_knowledge":
   - Non-obvious domain knowledge from this page that goes beyond the immediate goal — general patterns, related facts, taxonomies, or rules that would help answer SIMILAR questions in the future.
   - include genuinely informative knowledge that a general-purpose LLM is unlikely to already know. Skip common sense or textbook basics.
   - Leave empty string if nothing noteworthy.
   - If the page is a static structured reference (e.g. a lookup table, catalog, or index where all entries are directly readable from the page without any interaction), flag it as a reusable resource: note the URL and describe what kinds of queries it can answer.
   - Do NOT flag interactive pages (search forms, login-gated databases, JavaScript-rendered content) — they cannot be reliably re-fetched for different queries; instead, extract whatever information is already visible into "content" or "supplementary_knowledge" directly.


## Output (JSON)
{{"found": true/false, "content": "goal-relevant extracted information", "supplementary_knowledge": "non-obvious domain knowledge and useful resource patterns discovered on this page"}}"""


# ==================== HTML to Markdown ====================

class _LinkAwareMarkdownConverter(MarkdownConverter):
    """Custom Markdown converter that handles <a> tags without title."""

    def convert_a(self, el, text, parent_tags):
        if '_noformat' in parent_tags:
            return text
        prefix, suffix, text = chomp(text)
        if not text:
            text = ''
        href = el.get('href')
        title = el.get('title')
        if (self.options['autolinks']
                and text.replace(r'\_', '_') == href
                and not title
                and not self.options['default_title']):
            return '<%s>' % href
        if self.options['default_title'] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        return '%s[%s](%s%s)%s' % (prefix, text, href, title_part, suffix) if href else text


def _html_to_markdown(html: str, **options) -> str:
    return _LinkAwareMarkdownConverter(**options).convert(html)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import PyPDF2
    except ImportError:
        logger.warning("[ReadPage] PyPDF2 not installed, cannot extract PDF text")
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        logger.error(f"[ReadPage] PDF extraction failed: {e}")
        return ""


# ==================== Page Cache ====================

_page_cache: Dict[str, Tuple[Optional[str], str]] = {}


def clear_page_cache():
    """Clear the URL content cache. Call between questions."""
    _page_cache.clear()


def _normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _sharded_page_path(normalized_url: str) -> str:
    h = hashlib.md5(normalized_url.encode()).hexdigest()
    return os.path.join(RAW_PAGE_CACHE_DIR, h[:2], h + ".json")


def _load_cached_page(normalized_url: str) -> Optional[Tuple[Optional[str], str]]:
    if not RAW_PAGE_CACHE_ENABLED:
        return None
    def _read():
        path = _sharded_page_path(normalized_url)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        content = data.get("content") or None
        source = data.get("source", "")
        if source == "forbidden" and not content:
            return (None, "forbidden")
        return (content, source)
    try:
        result = _page_cache_io_pool.submit(_read).result(timeout=_PAGE_CACHE_IO_TIMEOUT)
        if result is not None:
            print(f"[ReadPage] cache hit: {normalized_url[:120]}", flush=True)
            logger.info(f"[ReadPage] cache hit: {normalized_url[:120]}")
        return result
    except Exception:
        return None


def _store_cached_page(normalized_url: str, original_url: str, content: str, source: str) -> None:
    if not RAW_PAGE_CACHE_ENABLED:
        return
    def _write():
        try:
            path = _sharded_page_path(normalized_url)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp.{os.getpid()}"
            data = {
                "normalized_url": normalized_url,
                "original_url": original_url,
                "content": content,
                "source": source,
                "updated_at": time.time(),
            }
            with open(tmp, "w") as f:
                json.dump(data, f, ensure_ascii=False)
            os.rename(tmp, path)
        except Exception as exc:
            logger.warning(f"[ReadPage] cache write failed: {exc}")
    _page_cache_io_pool.submit(_write)


# ==================== Page Fetch Functions ====================

def read_page_proxy(url: str, timeout: int = 20, max_retries: int = 3) -> Optional[str]:
    """Fetch page via search proxy service, returns Markdown content."""
    for attempt in range(max_retries):
        try:
            response = http_requests.post(
                PROXY_READ_URL,
                json={"url": url, "source": os.getenv("SEARCH_PROXY_READ_SOURCE", "")},
                timeout=timeout,
            )
            if response.status_code == 200 and response.json().get("code") == 200:
                reader_result = response.json()["data"]
                reader_html = brotli.decompress(base64.b64decode(reader_result["html"]))
                for encoding in ('utf-8', 'gbk', 'gb18030', 'latin-1'):
                    try:
                        html_text = reader_html.decode(encoding)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                else:
                    html_text = reader_html.decode('utf-8', errors='replace')
                md = _html_to_markdown(html_text).strip()
                return re.sub(r"\n{3,}", "\n\n", md)
            else:
                logger.warning(f"[ReadPage] proxy error (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.warning(f"[ReadPage] proxy exception (attempt {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(5 * attempt)
    return None


class _ForbiddenError(Exception):
    """Raised when a page returns HTTP 403 Forbidden."""
    pass


def read_page_requests(url: str, timeout: int = 15, max_retries: int = 2) -> Optional[str]:
    """Fetch page via direct HTTP request, returns Markdown content.

    Raises _ForbiddenError if the server returns 403 Forbidden.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    }
    for attempt in range(max_retries):
        try:
            response = http_requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 403:
                logger.warning(f"[ReadPage] 403 Forbidden: {url}")
                raise _ForbiddenError(f"403 Forbidden: {url}")
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if content_type == "application/pdf":
                md = _extract_pdf_text(response.content)
            elif content_type in ("text/html", "application/xhtml+xml"):
                md = _html_to_markdown(response.text).strip()
            elif content_type in (
                "text/plain", "text/css", "text/csv", "text/javascript",
                "application/xml", "text/xml", "application/json",
                "application/ld+json", "application/rtf",
            ):
                md = response.text
            else:
                logger.warning(f"[ReadPage] Unsupported content-type: {content_type}")
                return None

            return re.sub(r"\n{3,}", "\n\n", md)
        except _ForbiddenError:
            raise  # Don't retry 403, propagate immediately
        except Exception as e:
            logger.warning(f"[ReadPage] requests error (attempt {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(1)
    return None


def read_page_jina(
    url: str, api_key: str = "", timeout: int = 15, max_retries: int = 3
) -> Optional[str]:
    """Fetch page via Jina Reader API.

    Interface kept for future open-source use. Not enabled by default.
    """
    if not api_key:
        api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        logger.warning("[ReadPage] Jina API key not configured")
        return None

    for attempt in range(max_retries):
        try:
            response = http_requests.get(
                f"{JINA_READER_URL_PREFIX}{url}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            if response.status_code == 200 and response.text:
                return response.text
            logger.warning(
                f"[ReadPage] Jina error (attempt {attempt + 1}/{max_retries}): "
                f"status={response.status_code}"
            )
        except Exception as e:
            logger.warning(f"[ReadPage] Jina exception (attempt {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(1)
    return None


def read_page(
    url: str,
    mode: str = "proxy",
    timeout: int = 20,
    max_retries: int = 2,
    jina_api_key: str = "",
    max_content_length: int = 200000,
) -> Tuple[Optional[str], str]:
    """Unified page fetch entry with per-question caching.

    Successfully fetched pages are cached by URL. Failed fetches are NOT cached
    so that retries at higher levels can try again. Cache is cleared between
    questions via clear_page_cache().

    Args:
        url: Page URL.
        mode: Fetch mode - 'proxy' (default), 'requests', 'jina'.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
        jina_api_key: Jina API key (only for mode='jina').
        max_content_length: Max content length before truncation.

    Returns:
        (content, source) tuple. source is 'proxy'/'requests'/'jina'/'failed'.
    """
    normalized_url = _normalize_url(url)

    if normalized_url in _page_cache:
        cached = _page_cache[normalized_url]
        print(f"[ReadPage] memory cache hit: {normalized_url[:120]}", flush=True)
        logger.info(f"[ReadPage] memory cache hit: {normalized_url[:120]}")
        return cached

    cached = _load_cached_page(normalized_url)
    if cached is not None:
        _page_cache[normalized_url] = cached
        return cached

    result = _read_page_uncached(url, mode, timeout, max_retries, jina_api_key, max_content_length)

    if result[0] is not None:
        _page_cache[normalized_url] = result
        _store_cached_page(normalized_url, url, result[0], result[1])
    elif result[1] == "forbidden":
        # Cache forbidden pages so they are not retried
        forbidden_cached = (None, "forbidden")
        _page_cache[normalized_url] = forbidden_cached
        _store_cached_page(normalized_url, url, "", "forbidden")

    return result


def _read_page_uncached(
    url: str,
    mode: str = "proxy",
    timeout: int = 20,
    max_retries: int = 2,
    jina_api_key: str = "",
    max_content_length: int = 200000,
) -> Tuple[Optional[str], str]:
    """Actual fetch logic without caching."""
    if mode == "proxy":
        content = read_page_proxy(url, timeout, max_retries)
        if content:
            return _truncate_content(content, max_content_length), "proxy"
        # Fallback to direct HTTP
        try:
            content = read_page_requests(url, timeout, max_retries)
        except _ForbiddenError:
            return None, "forbidden"
        if content:
            return _truncate_content(content, max_content_length), "requests"
        return None, "failed"

    elif mode == "requests":
        try:
            content = read_page_requests(url, timeout, max_retries)
        except _ForbiddenError:
            return None, "forbidden"
        return (_truncate_content(content, max_content_length), "requests") if content else (None, "failed")

    elif mode == "jina":
        content = read_page_jina(url, jina_api_key, timeout, max_retries)
        return (_truncate_content(content, max_content_length), "jina") if content else (None, "failed")

    else:
        raise ValueError(f"Unknown fetch mode: {mode}")


# ==================== WebVisitTool ====================

class WebVisitTool:
    """Fetch web pages and extract goal-relevant information using a compression LLM.

    Two-stage pipeline:
    1. Fetch: retrieve raw page content (via proxy/requests/Jina)
    2. Compress: use a cheap LLM to extract only the goal-relevant information
    """

    # Global concurrency limiter for page fetching across all instances.
    # Controls the total number of concurrent page fetch requests, regardless
    # of how many WebVisitTool instances or questions are running in parallel.

    def __init__(
        self,
        compress_llm,
        mode: str = "proxy",
        fetch_timeout: int = 20,
        fetch_max_retries: int = 2,
        max_content_length: int = 200000,
        max_workers: int = 3,
        max_compress_context_tokens: int = 100000,
    ):
        """
        Args:
            compress_llm: LLMClient instance for content compression.
            mode: Fetch mode ('proxy', 'requests', 'jina').
            fetch_timeout: Timeout for page fetch (seconds).
            fetch_max_retries: Max retries for page fetch.
            max_content_length: Max raw content length before truncation.
            max_workers: Max threads for parallel page fetching.
            max_compress_context_tokens: Max estimated tokens for the content
                portion sent to the compression LLM. The prompt template and
                expected output tokens are reserved on top of this limit.
                Default 100000 works for models with ~120K context window.
        """
        self.compress_llm = compress_llm
        self.mode = mode
        self.fetch_timeout = fetch_timeout
        self.fetch_max_retries = fetch_max_retries
        self.max_content_length = max_content_length
        self.max_workers = max_workers
        self.max_compress_context_tokens = max_compress_context_tokens
        # Timing accumulators (reset per visit() call)
        self._fetch_time: float = 0.0
        self._compress_time: float = 0.0

    @classmethod
    def from_config(cls, config, compress_llm) -> "WebVisitTool":
        """Create from config dict.

        Args:
            config: Full config dict (must contain 'web_visit' section).
            compress_llm: LLMClient instance for compression.
        """
        wv = config.get("web_visit", {})
        return cls(
            compress_llm=compress_llm,
            mode=wv.get("fetch_mode", "proxy"),
            fetch_timeout=wv.get("fetch_timeout", 20),
            fetch_max_retries=wv.get("fetch_max_retries", 2),
            max_content_length=wv.get("max_content_length", 200000),
            max_compress_context_tokens=wv.get("max_compress_context_tokens", 100000),
        )

    async def visit(self, urls: List[str], goal: str) -> List[Dict]:
        """Fetch pages and extract goal-relevant information.

        Args:
            urls: List of URLs to visit.
            goal: The information goal (what to extract from pages).

        Returns:
            List of dicts, each containing:
                - url: str
                - fetch_source: str ('proxy'/'requests'/'jina'/'failed')
                - content_length: int (raw page content length)
                - found: bool (whether relevant info was found)
                - extracted_content: str (extracted information)
        """
        self._fetch_time = 0.0
        self._compress_time = 0.0

        async def _fetch_one_and_process(url: str) -> Dict:
            # Fetch (sync IO in thread) + compress
            t_fetch_start = time.time()
            content, source = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: read_page(
                    url,
                    mode=self.mode,
                    timeout=self.fetch_timeout,
                    max_retries=self.fetch_max_retries,
                    max_content_length=self.max_content_length,
                ),
            )
            self._fetch_time += time.time() - t_fetch_start
            page = {
                "url": url,
                "content": content,
                "fetch_source": source,
                "content_length": len(content) if content else 0,
            }
            return await self._process_page(page, goal)

        tasks = [_fetch_one_and_process(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def _process_page(self, page: Dict, goal: str) -> Dict:
        """Process a single fetched page: compress if content available."""
        if not page.get("content"):
            return {
                "url": page["url"],
                "fetch_source": page["fetch_source"],
                "content_length": 0,
                "found": False,
                "extracted_content": "",
            }

        extracted = await self._compress(page["content"], goal)
        return {
            "url": page["url"],
            "fetch_source": page["fetch_source"],
            "content_length": page["content_length"],
            **extracted,
        }

    def _fetch_pages(self, urls: List[str]) -> List[Dict]:
        """Fetch multiple pages in parallel using thread pool."""
        def fetch_one(url: str) -> Dict:
            content, source = read_page(
                url,
                mode=self.mode,
                timeout=self.fetch_timeout,
                max_retries=self.fetch_max_retries,
                max_content_length=self.max_content_length,
            )
            return {
                "url": url,
                "content": content,
                "fetch_source": source,
                "content_length": len(content) if content else 0,
            }

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(fetch_one, urls))

    async def _compress(self, content: str, goal: str) -> Dict:
        """Use compression LLM to extract goal-relevant info from page content.

        Returns:
            {"found": bool, "extracted_content": str}
        """
        # Truncate if too long (character-level first pass)
        if len(content) > self.max_content_length:
            content = content[: self.max_content_length] + "\n\n[...content truncated...]"

        # Token-aware truncation: estimate tokens to avoid exceeding model context.
        # Conservative ratio: ~1.5 chars per token for Chinese/mixed content.
        # This prevents "input longer than context length" errors from the LLM API.
        CHARS_PER_TOKEN = 1.5
        max_content_chars = int(self.max_compress_context_tokens * CHARS_PER_TOKEN)
        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "\n\n[...content truncated for token limit...]"

        prompt = WEBPAGE_EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        t_compress_start = time.time()
        response = await self.compress_llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        self._compress_time += time.time() - t_compress_start

        if not response:
            return {"found": False, "extracted_content": ""}

        return self._parse_extraction_result(response)

    @staticmethod
    def _parse_extraction_result(response: str) -> Dict:
        """Parse LLM extraction response into structured result."""
        def _coerce_text(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (dict, list)):
                try:
                    return json.dumps(value, ensure_ascii=False)
                except TypeError:
                    return str(value)
            return str(value).strip()

        def _build_result(parsed: Dict) -> Dict:
            return {
                "found": parsed.get("found", False),
                "extracted_content": _coerce_text(parsed.get("content", "")),
                "supplementary_knowledge": _coerce_text(parsed.get("supplementary_knowledge", "")),
            }

        # Try direct JSON parse
        try:
            parsed = json.loads(response)
            return _build_result(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in response
        left = response.find('{')
        right = response.rfind('}')
        if left != -1 and right > left:
            try:
                parsed = json.loads(response[left: right + 1])
                return _build_result(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: treat entire response as content
        return {
            "found": True,
            "extracted_content": response.strip(),
            "supplementary_knowledge": "",
        }
