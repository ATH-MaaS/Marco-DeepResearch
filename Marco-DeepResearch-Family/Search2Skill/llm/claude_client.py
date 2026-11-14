"""
Claude (Anthropic) LLM client.

Uses the Anthropic Messages API.  Set the following environment variables
or pass them explicitly:

    ANTHROPIC_BASE_URL  -- API base URL  (default: https://api.anthropic.com)
    ANTHROPIC_AUTH_TOKEN -- API key (or pass api_key= to call_claude)
"""

import asyncio
import logging
import os
import random
import uuid
from typing import List, Optional, Dict

import requests


ANTHROPIC_BASE_URL = os.getenv(
    "ANTHROPIC_BASE_URL",
    "https://api.anthropic.com",
).rstrip("/")

ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")

logger = logging.getLogger(__name__)


def _build_payload(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    system_text = None
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": api_messages,
        "temperature": temperature,
    }
    if system_text:
        payload["system"] = system_text

    return payload


def _build_headers(api_key: Optional[str] = None) -> dict:
    token = api_key or ANTHROPIC_AUTH_TOKEN
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": token or "",
    }
    return headers


async def call_claude(
    messages: List[Dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    max_retries: int = 6,
    timeout: float = 120.0,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    base_url = (api_base or ANTHROPIC_BASE_URL).rstrip("/")
    url = f"{base_url}/v1/messages"
    headers = _build_headers(api_key)
    payload = _build_payload(messages, model, temperature, max_tokens)

    current_retry = 0
    while True:
        try:
            resp = await asyncio.to_thread(
                lambda: requests.post(url, headers=headers, json=payload, timeout=timeout)
            )

            if resp.status_code == 429:
                current_retry += 1
                if current_retry > max_retries:
                    print(f"[Claude] {model} RateLimit exceeded after {max_retries} retries")
                    return None
                sleep_time = min(5 * current_retry + random.uniform(0, 5), 20)
                print(f"[Claude] {model} RateLimit, wait {sleep_time:.1f}s, retry {current_retry}/{max_retries}")
                await asyncio.sleep(sleep_time)
                continue

            if resp.status_code >= 500:
                current_retry += 1
                if current_retry > max_retries:
                    print(f"[Claude] {model} Server error {resp.status_code} after {max_retries} retries")
                    return None
                sleep_time = 2 * current_retry + random.uniform(0, 1)
                print(f"[Claude] {model} Server error {resp.status_code}, retry {current_retry}/{max_retries} in {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                continue

            if resp.status_code != 200:
                print(f"[Claude] {model} Unrecoverable error: {resp.status_code}\n{resp.text}")
                return None

            data = resp.json()
            texts = []
            for item in data.get("content", []):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts)

        except requests.exceptions.Timeout:
            current_retry += 1
            if current_retry > max_retries:
                print(f"[Claude] {model} Timeout after {max_retries} retries")
                return None
            sleep_time = 2 * current_retry + random.uniform(0, 1)
            print(f"[Claude] {model} Timeout, retry {current_retry}/{max_retries} in {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)

        except requests.exceptions.ConnectionError:
            current_retry += 1
            if current_retry > max_retries:
                print(f"[Claude] {model} Connection error after {max_retries} retries")
                return None
            sleep_time = 2 * current_retry + random.uniform(0, 1)
            print(f"[Claude] {model} Connection error, retry {current_retry}/{max_retries} in {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)

        except Exception as e:
            print(f"[Claude] {model} Unknown error: {e}")
            return None
