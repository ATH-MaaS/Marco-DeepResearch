import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI

from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__file__)

def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)

async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        try:
            server_port, sock = get_free_port(server_address)
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server.should_exit = True
            await server.serve()
            server_task = asyncio.create_task(server.main_loop())
            break
        except (OSError, SystemExit) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task

async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item
