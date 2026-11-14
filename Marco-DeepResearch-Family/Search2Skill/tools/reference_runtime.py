from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from llm.client import LLMClient
from tools.web_search import WebSearchTool
from tools.web_visit import WebVisitTool


REFERENCE_EXTRACTION_PROMPT = """Extract the requested reference information.

## Content
{content}

## Extraction Query
{extraction_query}

## Desired Output Format
{output_format}

## Instructions
1. Extract only the information needed to satisfy the extraction query.
2. Keep the result concise and directly usable by downstream code.
3. If the information is missing or unsupported by the content, return found=false.

## Output (JSON)
{{"found": true/false, "content": "short extracted result"}}
"""


class ReferenceRuntime:
    """Runtime helpers available in <CODE> code execution."""

    def __init__(self, config: Dict):
        self.config = config
        self._compress_llm: Optional[LLMClient] = None
        self._web_visit: Optional[WebVisitTool] = None

    def fetch_pages(self, urls: List[str], max_attempts: int = 3) -> str:
        """Fetch raw page content with the configured fetch mode, with retry."""
        urls = self._normalize_urls(urls)
        if not urls:
            return ""
        max_chars = self.config.get("web_visit", {}).get("max_content_length", 200000)

        for attempt in range(1, max_attempts + 1):
            raw_pages = self._get_web_visit()._fetch_pages(urls)
            parts = []
            for page in raw_pages:
                content = (page.get("content") or "").strip()
                if not content:
                    continue
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n...[truncated]"
                parts.append(f"[URL] {page.get('url', '')}\n{content}")
            if parts:
                return "\n\n---\n\n".join(parts)
            if attempt < max_attempts:
                import time
                wait = 5 * attempt
                print(
                    f"[ReferenceRuntime] All pages failed to fetch, "
                    f"retrying {attempt}/{max_attempts} in {wait}s..."
                )
                time.sleep(wait)

        return ""

    def ask_llm_to_extract(
        self,
        content: str,
        extraction_query: str,
        output_format: str = "",
    ) -> Dict:
        """Use the configured compression LLM to extract concise reference info."""
        return self._run(self._ask_llm_to_extract_async(content, extraction_query, output_format))

    def fetch_and_extract_reference(
        self,
        urls: List[str],
        extraction_query: str,
        output_format: str = "",
    ) -> Dict:
        """Recommended helper: fetch configured URLs and extract requested info."""
        return self._run(
            self._fetch_and_extract_reference_async(urls, extraction_query, output_format)
        )

    async def _ask_llm_to_extract_async(
        self,
        content: str,
        extraction_query: str,
        output_format: str = "",
    ) -> Dict:
        if not content.strip():
            return {"found": False, "content": ""}

        max_chars = self.config.get("web_visit", {}).get("max_content_length", 200000)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[...content truncated...]"

        prompt = REFERENCE_EXTRACTION_PROMPT.format(
            content=content,
            extraction_query=extraction_query.strip() or "(empty)",
            output_format=output_format.strip() or "Return a concise directly usable result.",
        )

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            response = await self._get_compress_llm().chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            if response is not None:
                return self._parse_extract_response(response)
            if attempt < max_attempts:
                wait = 5 * attempt
                logger.warning(
                    f"[ReferenceRuntime] LLM returned None, "
                    f"retrying {attempt}/{max_attempts} in {wait}s"
                )
                print(
                    f"[ReferenceRuntime] LLM extraction failed, "
                    f"retrying {attempt}/{max_attempts} in {wait}s..."
                )
                await asyncio.sleep(wait)

        logger.warning("[ReferenceRuntime] LLM extraction failed after all attempts")
        return {"found": False, "content": ""}

    async def _fetch_and_extract_reference_async(
        self,
        urls: List[str],
        extraction_query: str,
        output_format: str = "",
    ) -> Dict:
        urls = self._normalize_urls(urls)
        if not urls:
            return {"found": False, "content": ""}

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            result = await self._fetch_and_extract_once(
                urls, extraction_query, output_format
            )
            if result.get("found") or result.get("content"):
                return result
            if attempt < max_attempts:
                wait = 10
                print(
                    f"[ReferenceRuntime] fetch_and_extract got no result, "
                    f"retrying whole pipeline ({attempt}/{max_attempts}) in {wait}s..."
                )
                await asyncio.sleep(wait)

        return result

    async def _fetch_and_extract_once(
        self,
        urls: List[str],
        extraction_query: str,
        output_format: str = "",
    ) -> Dict:
        goal = self._compose_goal(extraction_query, output_format)
        visit_results = await self._get_web_visit().visit(urls, goal)
        extracted_parts = []
        fetched_any = False
        for item in visit_results:
            if item.get("content_length", 0) > 0:
                fetched_any = True
            extracted = (item.get("extracted_content") or "").strip()
            if not extracted:
                continue
            extracted_parts.append(f"[URL] {item.get('url', '')}\n{extracted}")

        if not extracted_parts:
            if not fetched_any:
                # Pages themselves failed to fetch — wait before fallback retry
                print("[ReferenceRuntime] Visit fetched no pages, waiting 5s before fallback fetch...")
                await asyncio.sleep(5)
            raw_content = self.fetch_pages(urls)
            return await self._ask_llm_to_extract_async(raw_content, extraction_query, output_format)

        combined = "\n\n---\n\n".join(extracted_parts)
        return await self._ask_llm_to_extract_async(combined, extraction_query, output_format)

    def _compose_goal(self, extraction_query: str, output_format: str = "") -> str:
        query = extraction_query.strip() or "(empty)"
        output_hint = output_format.strip()
        return query if not output_hint else f"{query}\nDesired output format: {output_hint}"

    def _normalize_urls(self, urls) -> List[str]:
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list):
            return []
        out = []
        for url in urls:
            text = str(url or "").strip()
            if text:
                out.append(text)
        return out

    def _get_compress_llm(self) -> LLMClient:
        if self._compress_llm is None:
            profile = self.config.get("web_visit", {}).get("compress_llm", "qwen-turbo")
            self._compress_llm = LLMClient.from_config(self.config, profile_name=profile)
        return self._compress_llm

    def _get_web_visit(self) -> WebVisitTool:
        if self._web_visit is None:
            self._web_visit = WebVisitTool.from_config(
                self.config,
                compress_llm=self._get_compress_llm(),
            )
        return self._web_visit

    def _run(self, coro):
        return asyncio.run(coro)

    @staticmethod
    def _parse_extract_response(response: Optional[str]) -> Dict:
        if not response:
            return {"found": False, "content": ""}

        try:
            parsed = json.loads(response)
            return {
                "found": bool(parsed.get("found", False)),
                "content": str(parsed.get("content", "")).strip(),
            }
        except Exception:
            pass

        left = response.find("{")
        right = response.rfind("}")
        if left != -1 and right > left:
            try:
                parsed = json.loads(response[left:right + 1])
                return {
                    "found": bool(parsed.get("found", False)),
                    "content": str(parsed.get("content", "")).strip(),
                }
            except Exception:
                pass

        text = response.strip()
        return {"found": bool(text), "content": text}
