import asyncio
import logging
import os
import random
import subprocess
import socket
import time
from datetime import datetime
from pathlib import Path
from openai import (
    AsyncOpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    AuthenticationError,
    BadRequestError,
)
from typing import List, Optional, Union, Dict

from llm.claude_client import call_claude


def _is_claude_model(model: str) -> bool:
    return "claude" in model.lower()


def _classify_llm_error(error: Exception) -> str:
    message = str(error).lower()
    context_markers = (
        "maximum context length",
        "context length",
        "input_tokens",
        "too many tokens",
        "reduce the length of the input",
        "prompt is too long",
        "context window",
    )
    if any(marker in message for marker in context_markers):
        return "context_overflow"
    return type(error).__name__


def _get_llm_traj_logger():
    if os.environ.get("EXSKILL_LLM_TRAJ_LOGGING", "false").lower() not in {"1", "true", "yes", "on"}:
        return None

    logger = logging.getLogger(f"llm_trajectory.{os.getpid()}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = Path("results") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_pid{os.getpid()}_llm_traj.log"
        handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def _start_vllm(model_path: str, port: int = 8000,
                 gpu_memory_utilization: float = 0.9,
                 tensor_parallel_size: int = 1) -> subprocess.Popen:
    """Start a vLLM server if not already running on the port."""
    if _port_in_use(port):
        print(f"[vLLM] Port {port} already in use, assuming vLLM is running")
        return None

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
    ]
    print(f"[vLLM] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)

    # Wait for server to be ready
    for i in range(120):
        if _port_in_use(port):
            print(f"[vLLM] Server ready on port {port}")
            return proc
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"[vLLM] Server exited with code {proc.returncode}")

    proc.kill()
    raise RuntimeError(f"[vLLM] Server failed to start within 120s")


class LLMClient:
    """LLM client compatible with vLLM and any OpenAI-compatible API."""

    _vllm_procs = {}  # port -> Popen, avoid duplicate launches

    def __init__(self, api_base: str, api_key: str, model: str = "qwen3-8b",
                 temperature: float = 0.3, max_tokens: int = 4096,
                 timeout: float = 120.0, max_retries: int = 6,
                 profile_name: Optional[str] = None,
                 enable_thinking: bool = False,
                 stream: Optional[bool] = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
        )
        self.api_base = api_base
        self.api_key = api_key
        self.default_model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.max_retries = max_retries
        self.profile_name = profile_name or ""
        self.enable_thinking = enable_thinking
        self.stream = stream
        model_name = (Path(model).name if isinstance(model, str) else str(model or "")).lower()
        profile_name_lc = self.profile_name.lower()
        self._is_qwen = "qwen" in profile_name_lc or "qwen" in model_name
        self._api_callback = None
        self.last_error_type = None
        self.last_error_message = None


    @classmethod
    def from_config(cls, config, profile_name: str = None) -> "LLMClient":
        """Create LLMClient from config.

        Supports three formats:
        1. Multi-profile: config["llm"]["profiles"][name] + config["llm"]["default"]
        2. Single config: config["llm"] directly contains api_base/model fields
        3. Local vLLM: profile contains backend: "vllm" + model_path, auto-starts vLLM
        """
        llm_config = config["llm"]

        if "profiles" in llm_config:
            name = profile_name or llm_config.get("default")
            profiles = llm_config["profiles"]
            if name not in profiles:
                available = list(profiles.keys())
                raise ValueError(f"LLM profile '{name}' not found. Available: {available}")
            profile = profiles[name]
        else:
            profile = llm_config

        # vLLM backend: auto-start server
        if profile.get("backend") == "vllm":
            model_path = profile["model_path"]
            port = profile.get("port", 8000)

            if port not in cls._vllm_procs:
                proc = _start_vllm(
                    model_path=model_path,
                    port=port,
                    gpu_memory_utilization=profile.get("gpu_memory_utilization", 0.9),
                    tensor_parallel_size=profile.get("tensor_parallel_size", 1),
                )
                if proc:
                    cls._vllm_procs[port] = proc

            return cls(
                api_base=f"http://localhost:{port}/v1",
                api_key="empty",
                model=model_path,
                temperature=profile.get("temperature", 0.3),
                max_tokens=profile.get("max_tokens", 16384),
                timeout=profile.get("timeout", 200.0),
                max_retries=profile.get("max_retries", 5),
                profile_name=name,
                enable_thinking=profile.get("enable_thinking", False),
                stream=profile.get("stream"),
            )

        return cls(
            api_base=profile["api_base"],
            api_key=profile["api_key"],
            model=profile["model"],
            temperature=profile.get("temperature", 0.7),
            max_tokens=profile.get("max_tokens", 16384),
            timeout=profile.get("timeout", 120.0),
            max_retries=profile.get("max_retries", 5),
            profile_name=name if "profiles" in llm_config else "",
            enable_thinking=profile.get("enable_thinking", False),
            stream=profile.get("stream"),
        )

    def set_api_callback(self, callback):
        """Set a callback to monitor API call outcomes.

        Args:
            callback: callable(profile_name, model, success) called once per
                request with the final outcome (success=True after all retries
                succeed, success=False when ultimately failing).
        """
        self._api_callback = callback

    async def chat(self,
                   messages: Union[List[Dict], str],
                   temperature: float = None,
                   model: str = None,
                   max_tokens: int = None,
                   stop: Optional[List[str]] = None) -> Optional[str]:
        """Core chat interface with exponential backoff retry."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        use_model = model or self.default_model
        use_temp = temperature if temperature is not None else self.default_temperature
        use_max_tokens = max_tokens or self.default_max_tokens
        self.last_error_type = None
        self.last_error_message = None

        if _is_claude_model(use_model):
            result = await call_claude(
                messages=messages,
                model=use_model,
                temperature=use_temp,
                max_tokens=use_max_tokens,
                max_retries=self.max_retries,
                timeout=self.client.timeout,
                api_base=self.api_base,
                api_key=self.api_key,
            )
            llm_traj_logger = _get_llm_traj_logger()
            if llm_traj_logger is not None:
                llm_traj_logger.info("\n\n========= LLM Chat (Claude) ===============")
                llm_traj_logger.info(f"model: {use_model}")
                llm_traj_logger.info(f"history_messages: {messages}")
                llm_traj_logger.info(f"LLM Response: {result}")
            if self._api_callback:
                self._api_callback(
                    profile_name=self.profile_name,
                    model=use_model,
                    success=result is not None,
                )
            if result is not None:
                self.last_error_type = None
                self.last_error_message = None
            return result

        current_retry = 0
        while True:
            try:
                use_model_name = (Path(use_model).name if isinstance(use_model, str) else str(use_model or "")).lower()
                is_qwen_call = self._is_qwen or "qwen" in use_model_name
                extra_body = {}
                if is_qwen_call:
                    extra_body["enable_thinking"] = self.enable_thinking
                    extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

                create_kwargs = dict(
                    model=use_model,
                    messages=messages,
                    temperature=use_temp,
                    stop=stop,
                    extra_body=extra_body,
                )
                if use_max_tokens is not None:
                    create_kwargs["max_tokens"] = use_max_tokens
                if self.stream is True:
                    create_kwargs["stream"] = True

                if self.stream is True:
                    chunks = []
                    reasoning_chunks = []
                    async for chunk in await self.client.chat.completions.create(**create_kwargs):
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta:
                            if delta.content:
                                chunks.append(delta.content)
                            rc = getattr(delta, 'reasoning_content', None)
                            if rc:
                                reasoning_chunks.append(rc)
                    reasoning = "".join(reasoning_chunks)
                    answer = "".join(chunks)
                    if reasoning:
                        content = "<think>\n" + reasoning + "\n</think>\n" + answer
                    else:
                        content = answer
                else:
                    response = await self.client.chat.completions.create(**create_kwargs)
                    content = response.choices[0].message.content
                llm_traj_logger = _get_llm_traj_logger()
                if llm_traj_logger is not None:
                    llm_traj_logger.info("\n\n========= LLM Chat ===============")
                    llm_traj_logger.info(f"model: {use_model}")
                    llm_traj_logger.info(f"history_messages: {messages}")
                    llm_traj_logger.info(f"LLM Response: {content}")
                if self._api_callback:
                    self._api_callback(profile_name=self.profile_name, model=use_model, success=True)
                self.last_error_type = None
                self.last_error_message = None
                return content

            except RateLimitError:
                current_retry += 1
                if current_retry > self.max_retries:
                    print(f"[LLM] {use_model} RateLimit exceeded after {self.max_retries} retries")
                    self.last_error_type = "RateLimitError"
                    self.last_error_message = "RateLimit exceeded after retries"
                    if self._api_callback:
                        self._api_callback(profile_name=self.profile_name, model=use_model, success=False)
                    return None
                sleep_time = min(5 * current_retry + random.uniform(0, 5), 20)
                print(f"[LLM] {use_model} TPM limit, waiting {sleep_time:.1f}s before retry {current_retry}/{self.max_retries}")
                await asyncio.sleep(sleep_time)

            except (APITimeoutError, APIConnectionError, InternalServerError) as e:
                current_retry += 1
                if current_retry > self.max_retries:
                    print(f"[LLM {use_model}] Network error after {self.max_retries} retries: {e}")
                    self.last_error_type = type(e).__name__
                    self.last_error_message = str(e)
                    if self._api_callback:
                        self._api_callback(profile_name=self.profile_name, model=use_model, success=False)
                    return None
                if current_retry == 3:
                    print(
                        f"[LLM] {use_model} Retry threshold reached (3/{self.max_retries}). "
                        f"api_base={self.client.base_url} "
                        f"Dumping request payload for debugging:",
                    )
                    print(
                        {
                            "model": use_model,
                            "api_base": str(self.client.base_url),
                            "temperature": use_temp,
                            "max_tokens": use_max_tokens,
                            "stop": stop,
                            "enable_thinking": self.enable_thinking,
                            "messages": messages,
                        }
                    )
                sleep_time = 2 * current_retry + random.uniform(0, 1)
                print(f"[LLM] {use_model} Network issue, retry {current_retry}/{self.max_retries} in {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

            except (AuthenticationError, BadRequestError) as e:
                print(f"[LLM] {use_model} Unrecoverable error: {e}")
                self.last_error_type = _classify_llm_error(e)
                self.last_error_message = str(e)
                request_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
                print(
                    f"[LLM] classified_error={self.last_error_type} "
                    f"messages={len(messages)} prompt_chars={request_chars} "
                    f"max_tokens={use_max_tokens}"
                )
                llm_traj_logger = _get_llm_traj_logger()
                if llm_traj_logger is not None:
                    llm_traj_logger.info("\n\n========= LLM Chat FAILED ===============")
                    llm_traj_logger.info(f"model: {use_model}")
                    llm_traj_logger.info(f"error: {e}")
                    llm_traj_logger.info(f"history_messages: {messages}")
                if self._api_callback:
                    self._api_callback(profile_name=self.profile_name, model=use_model, success=False)
                return None

            except Exception as e:
                print(f"[LLM] {use_model} Unknown error: {e}")
                self.last_error_type = _classify_llm_error(e)
                self.last_error_message = str(e)
                if self._api_callback:
                    self._api_callback(profile_name=self.profile_name, model=use_model, success=False)
                return None
