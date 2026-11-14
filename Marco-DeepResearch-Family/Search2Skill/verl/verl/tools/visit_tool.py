import os
import json
import requests
import traceback
import io
from typing import Any, Optional, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from openai import OpenAI
from markdownify import MarkdownConverter, chomp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from verl.utils.rollout_trace import rollout_trace_op

# Prompt definition (was imported from src.prompt)
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", "150000"))
JINA_READER_URL_PREFIX = "https://r.jina.ai/"

class JinaKeyManager:
    """Jina API Key Manager for sequential key rotation"""
    def __init__(self):
        self.current_index = 0
    
    def _get_keys_from_env(self):
        jina_api_key = os.getenv("JINA_API_KEY")
        jina_api_keys_str = os.getenv("JINA_API_KEYS", "")
        
        jina_api_keys = [key.strip() for key in jina_api_keys_str.split("\n") if key.strip()]
        
        if jina_api_key and jina_api_key not in jina_api_keys:
            jina_api_keys.insert(0, jina_api_key)
        
        return jina_api_keys if jina_api_keys else [jina_api_key] if jina_api_key else []
    
    def get_current_key(self):
        keys = self._get_keys_from_env()
        if not keys:
            return None
        return keys[self.current_index % len(keys)]
    
    def switch_to_next_key(self):
        keys = self._get_keys_from_env()
        if keys:
            self.current_index = (self.current_index + 1) % len(keys)
            jina_current_key = keys[self.current_index]
            print(f"Jina key switch to: {jina_current_key}")

# Global key manager instance
jina_key_manager = JinaKeyManager()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from a PDF file."""
    try:
        import PyPDF2
    except ImportError as e:
        raise ImportError(
            "You must install package `PyPDF2` to run this tool: for instance run `pip install PyPDF2`."
        ) from e

    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

class MyMarkdownConverter(MarkdownConverter):
    """ Wrapper class for Markdownify's MarkdownConverter. Converts <a> tags without titles """
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

def my_markdownify(html, **options):
    return MyMarkdownConverter(**options).convert(html)

class VisitTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.openai_api_key = config.get('openai_api_key', os.getenv('SUMMARY_MODEL_OPENAI_API_KEY', ''))
        self.openai_api_base = config.get('openai_api_base', os.getenv('SUMMARY_MODEL_OPENAI_BASE_URL', ''))
        self.model_name = config.get('model_name', os.getenv('SUMMARY_MODEL_NAME', 'qwen-turbo'))
        self.visit_mode = config.get('visit_mode', os.getenv("VISIT_MODE", "requests"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_server(self, msgs):
        client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        # Removing manual retry loop here as we use tenacity
        try:
            chat_response = client.chat.completions.create(
                model=self.model_name,
                messages=msgs,
                stop=["\n<tool_response>", "<tool_response>"],
                temperature=0.7
            )
            content = chat_response.choices[0].message.content
            if content:
                try:
                    json.loads(content)
                except:
                    # extract json from string
                    left = content.find('{')
                    right = content.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        content = content[left:right+1]
                return content
        except Exception as e:
            print(f"Call summary LLM server error in tool_visit: {e}")
            raise # Re-raise to let tenacity handle retry
        return ""

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), 
           retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError)))
    def requests_readpage(self, url: str, timeout: int = 15) -> str:
        # Removing manual retry logic here as we use tenacity
        try:
            import re
            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
             return "[visit] Failed to load required packages."

        headers = {
            "User-Agent": (
                "Mozilla/5.0 Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            file_type = response.headers.get("content-type", "").split(";")[0].strip()
            if file_type == "application/pdf":
                markdown_content = extract_text_from_pdf(io.BytesIO(response.content))
            elif file_type == "text/html":
                markdown_content = my_markdownify(response.text).strip()
            elif file_type in ["text/plain", "application/json"]: 
                markdown_content = response.text
            else:
                # Default to text
                markdown_content = response.text

            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            print(f"Web content length: {len(markdown_content)}, truncate lenght: {WEBCONTENT_MAXLENGTH}")
            return self._truncate_content(markdown_content, WEBCONTENT_MAXLENGTH)

        except Exception as e:
            # Check if it's a NameResolutionError (DNS issue), often fatal, no point retrying if domain doesn't exist
            if "Name or service not known" in str(e) or "NameResolutionError" in str(e):
                print(f"Tool:[visit][requests] Fatal DNS error for {url}: {e}")
                raise e # Will be caught by outer try-except in readpage or propagate
            
            # Specifically for HTTP 403 Forbidden (e.g., anti-bot protection), retrying might work if it's temporary or UA/cookie related, 
            # but often it's persistent. However, since user asked if it covers it, we ensure HTTPError is in retry types.
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 403:
                 print(f"Tool:[visit][requests] 403 Forbidden for {url}: {e}")
                 # We let it retry in case it's a transient rate limit, but log it.
            
            message = f"An error occurred: {e}\nTraceback details:\n{traceback.format_exc()}"
            print(f"Tool:[visit][requests] Exception message: {message}")
            raise # Re-raise to let tenacity handle retry

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def jina_readpage(self, url: str) -> str:
        # Removing manual retry logic here as we use tenacity
        timeout = 15
        
        current_key = jina_key_manager.get_current_key()
        if not current_key:
            print("[visit] Jina API key not configured.")
            return "[visit] Failed to read page."
        
        headers = {"Authorization": f"Bearer {current_key}"}
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 402:
                print(f"Jina API key quota exceeded (402), switching to next key")
                jina_key_manager.switch_to_next_key()
                # Retry immediately with new key if possible, or let tenacity handle it
                raise ValueError("Jina quota exceeded")
            else:
                raise ValueError("jina readpage error")
        except Exception as e:
            raise # Re-raise to let tenacity handle retry

    def readpage(self, url: str, goal: str, mode: str=None) -> str:
        if mode is None:
            mode = self.visit_mode
        
        max_attempts = 2
        def is_failed(content):
            return (not content or 
                    content.startswith("[visit] Failed to read page.") or 
                    content == "[visit] Empty content.")

        content = None
        service = None

        for attempt in range(max_attempts):
            if mode == "requests":
                try:
                    content = self.requests_readpage(url)
                except Exception as e:
                    if "NameResolutionError" in str(e) or "Name or service not known" in str(e):
                         return f"[visit] Failed to read page {url}: Domain name not found."
                    if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 403:
                         return f"[visit] Failed to read page {url}: Access forbidden (403)."
                    content = "[visit] Failed to read page."
                service = "requests"
            elif mode == "jina":
                try:
                    content = self.jina_readpage(url)
                except Exception:
                    content = "[visit] Failed to read page."
                service = "jina"
            elif mode == "hybrid":
                try:
                    content = self.requests_readpage(url)
                except Exception as e:
                    if "NameResolutionError" in str(e) or "Name or service not known" in str(e):
                         # If DNS fails, Jina likely fails too, but let's fail gracefully here or try Jina if Jina handles it differently
                         content = "[visit] Failed to read page." 
                    else:
                         content = "[visit] Failed to read page."
                service = "requests"
                if is_failed(content):
                    print("requests_readpage failed, trying jina_readpage")
                    try:
                        content = self.jina_readpage(url)
                    except Exception:
                        content = "[visit] Failed to read page."
                    service = "jina"
            
            if not is_failed(content):
                content = content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                
                summary_retries = 3
                try:
                    raw = self.call_server(messages)
                except Exception:
                    raw = ""
                
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    content = content[:truncate_length]
                    messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                    try:
                        raw = self.call_server(messages)
                    except Exception:
                        raw = ""
                    summary_retries -= 1

                # Try parsing result
                try:
                    parsed = json.loads(raw)
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(parsed.get("evidence", "")) + "\n\n"
                    useful_information += "Summary: \n" + str(parsed.get("summary", "")) + "\n\n"
                    return useful_information
                except:
                    # Fallback to raw text if JSON parsing fails
                    if len(raw) > 10:
                         return f"Summary of {url} for goal {goal}:\n{raw}"

            if attempt == max_attempts - 1:
                return f"[visit] Failed to read page {url}."

        return f"[visit] Failed to read page {url}."

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, Dict[str, Any]]:
        try:
            urls = parameters.get("urls")
            goal = parameters.get("goal")
            
            if not urls or not goal:
                 return ToolResponse(text="[Visit] Invalid request format: Input must contain 'urls' and 'goal' fields"), 0.0, {}

            if isinstance(urls, str):
                urls = [urls]
            
            response_list = []
            # Blocking executor for simplicity
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in urls}
                for future in as_completed(futures):
                    try:
                        response_list.append(future.result())
                    except Exception as e:
                         response_list.append(f"Error fetching {futures[future]}: {str(e)}")
            
            response_text = "\n=======\n".join(response_list)
            return ToolResponse(text=response_text), 0.0, {}
            
        except Exception as e:
            return ToolResponse(text=f"[Visit] Error occurred: {str(e)}"), 0.0, {"error": str(e)}
