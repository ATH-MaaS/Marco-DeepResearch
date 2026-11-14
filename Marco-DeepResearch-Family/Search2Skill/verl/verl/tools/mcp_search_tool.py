
import json
import logging
import os
import re

from verl.tools.mcp_base_tool import MCPBaseTool

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MCPSearchTool(MCPBaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def _parse_tool_result(self, content: list) -> tuple[str, dict]:
        res = ""
        res_cnt = 0
        query_list = []
        metadata = {
            "api_request_error": "",
            "status": "unknown",
            "total_results": 0,
        }
        try:
            for part in content:
                if part.type != "text":
                    continue
                text = part.text.replace("'", '"')
                query_match = re.search(r'query"\s*:\s*"([^"]+)"', text)
                query = query_match.group(1) if query_match else ""
                query_list.append(query)

                title_matches = re.findall(r'"title"\s*:', text)
                title_count = len(title_matches)

                results_match = re.search(r'"results"\s*:\s*(\[.*?\])', text, re.DOTALL)
                results_content = results_match.group(1) if results_match else ""

                res += results_content
                res_cnt += title_count
        except json.JSONDecodeError:
            err_msg = "json parse error."
            logger.error(err_msg)
            metadata["api_request_error"] = err_msg
            metadata["status"] = "error"

        # update metadata
        metadata["status"] = "success"
        metadata["queries"] = query_list
        metadata["query_count"] = len(query_list)
        metadata["total_results"] = res_cnt
        return res, metadata
