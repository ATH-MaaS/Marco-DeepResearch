"""LiveCodeBench benchmark adapter.

Loads competitive programming problems from HuggingFace cache.
Evaluates by running extracted code against public test cases via subprocess.
"""

import json
import os
import random
import re
import subprocess
import tempfile
from typing import Dict, List

from .base import BaseBenchmark


_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "livecode",
)

# EvoAgentBench split file (if available)
_EVOAGENTBENCH_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "evoagentbench",
)
_SPLIT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "livecode", "task_split.json",
)


class LiveCodeBenchBenchmark(BaseBenchmark):
    """LiveCodeBench: competitive programming problems with test-case-based evaluation."""

    @staticmethod
    def download_from_hf(cache_dir: str = _DEFAULT_CACHE_DIR,
                         release_version: str = "release_v6") -> str:
        """Download from HuggingFace and cache locally. Returns cache path."""
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{release_version}.json")
        if os.path.exists(cache_path):
            return cache_path
        print(f"[LiveCodeBench] Downloading from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            release_version,
            split="test",
            trust_remote_code=True,
        )
        records = [dict(row) for row in ds]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
        print(f"[LiveCodeBench] Cached {len(records)} problems to {cache_path}")
        return cache_path

    def load(self, data_dir: str, num_samples: int = -1,
             subset: str = None, seed: int = None) -> List[Dict]:
        """Load LiveCodeBench problems.

        data_dir: path to a JSON/JSONL cache file, or a directory containing it.
        subset: filter by difficulty (e.g. "easy", "medium", "hard").
        """
        # Resolve file path
        if os.path.isfile(data_dir):
            cache_path = data_dir
        elif os.path.isdir(data_dir):
            # Look for cached file
            for name in ["release_v6.json", "livecode.json", "livecode.jsonl"]:
                candidate = os.path.join(data_dir, name)
                if os.path.exists(candidate):
                    cache_path = candidate
                    break
            else:
                cache_path = self.download_from_hf(cache_dir=data_dir)
        else:
            # Assume it's a path that should be auto-downloaded
            cache_path = self.download_from_hf()

        with open(cache_path, "r", encoding="utf-8") as f:
            if cache_path.endswith(".jsonl"):
                records = [json.loads(line) for line in f if line.strip()]
            else:
                records = json.load(f)

        items: List[Dict] = []
        for rec in records:
            qid = str(rec.get("question_id", rec.get("name", len(items))))
            difficulty = rec.get("difficulty", "unknown")
            if isinstance(difficulty, dict):
                difficulty = difficulty.get("value", "unknown")
            difficulty = str(difficulty)

            # Parse public test cases
            public_tests = rec.get("public_test_cases", [])
            if isinstance(public_tests, str):
                try:
                    public_tests = json.loads(public_tests)
                except json.JSONDecodeError:
                    public_tests = []
            parsed_tests = []
            for t in public_tests:
                if isinstance(t, dict):
                    parsed_tests.append({
                        "input": t.get("input", ""),
                        "output": t.get("output", t.get("expected_output", "")),
                    })

            fn_name = ""
            meta = rec.get("metadata", {})
            if isinstance(meta, dict):
                fn_name = meta.get("func_name", "")

            if subset and difficulty.lower() != subset.lower():
                continue

            starter_code = rec.get("starter_code", "")

            items.append({
                "id": qid,
                "question": rec.get("question_content", ""),
                "options": [],
                "context": "",
                "ground_truth": json.dumps({
                    "public_tests": parsed_tests,
                    "starter_code": starter_code,
                    "fn_name": fn_name,
                }),
                "starter_code": starter_code,
                "public_tests": parsed_tests,
                "fn_name": fn_name,
                "domain": difficulty,
                "difficulty": difficulty,
                "metadata": {
                    "difficulty": difficulty,
                    "subfield": "competitive_programming",
                    "source": "livecodebench",
                },
            })

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)

        return items

    def load_split(self, cache_path: str = None, split: str = "test",
                   num_samples: int = -1, seed: int = None) -> List[Dict]:
        """Load items filtered by EvoAgentBench train/test split."""
        if not os.path.exists(_SPLIT_FILE):
            raise FileNotFoundError(f"Split file not found: {_SPLIT_FILE}")
        with open(_SPLIT_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)

        split_ids = set()
        for cluster in raw.get("clusters", {}).values():
            split_ids.update(cluster.get(split, []))

        all_items = self.load(cache_path or _DEFAULT_CACHE_DIR, num_samples=-1)
        items = [it for it in all_items if it["id"] in split_ids]

        if num_samples > 0 and num_samples < len(items):
            rng = random.Random(seed) if seed is not None else random.Random()
            items = rng.sample(items, num_samples)
        return items

    def format_question(self, item: Dict) -> str:
        """Format the problem as a prompt for the LLM."""
        starter = item.get("starter_code", "")
        question = item["question"]
        if starter:
            fmt = (
                "### Format\nComplete the starter code below. "
                "Enclose your solution in ```python``` delimiters.\n"
                f"```python\n{starter}\n```\n"
            )
            code_sandbox_note = ""
        else:
            fmt = (
                "### Format\nRead inputs from stdin, solve the problem, write to stdout. "
                "Enclose your code in ```python``` delimiters.\n"
            )
            # code_sandbox_note = (
            #     "\n\n### Code Execution Note\n"
            #     "When using <CODE> to test your solution, stdin is NOT available in the sandbox. "
            #     "You must hardcode sample inputs directly in your code for testing. For example:\n"
            #     "```python\n"
            #     "import sys\n"
            #     "from io import StringIO\n"
            #     "sys.stdin = StringIO('3 5\\n1 2 3\\n')  # paste sample input here\n"
            #     "```\n"
            #     "After verifying correctness with hardcoded inputs, put your final stdin-based "
            #     "solution in <ANSWER>."
            # )
            code_sandbox_note = ""
        return (
            f"Solve the following competitive programming problem.\n\n"
            f"### Question\n{question}\n\n{fmt}\n"
            f"Put your final solution in <ANSWER>```python\\nyour_code\\n```</ANSWER>."
            f"{code_sandbox_note}"
        )

    def get_ground_truth(self, item: Dict) -> str:
        """Return JSON-encoded test cases and metadata for evaluation."""
        return item["ground_truth"]

    def evaluate(self, prediction: str, ground_truth: str, **kwargs) -> Dict:
        """Run extracted code against test cases."""
        try:
            gt_data = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return {"correct": False, "score": 0.0, "detail": "Invalid ground_truth format"}

        tests = gt_data.get("public_tests", [])
        if not tests:
            return {"correct": False, "score": 0.0, "detail": "No test cases"}

        code = self._extract_code(prediction)
        if not code:
            return {"correct": False, "score": 0.0, "detail": "No code extracted"}

        starter_code = gt_data.get("starter_code", "")
        is_function_style = bool(starter_code and starter_code.strip())

        passed = 0
        total = len(tests)
        details = []
        for i, tc in enumerate(tests):
            inp = tc.get("input", "")
            expected = tc.get("output", "").strip()
            try:
                if is_function_style:
                    actual = self._run_function_code(code, inp, starter_code).strip()
                else:
                    actual = self._run_code(code, inp).strip()
                if actual == expected:
                    passed += 1
                    details.append(f"Test {i+1}: PASS")
                else:
                    details.append(
                        f"Test {i+1}: FAIL (expect={expected[:100]!r}, got={actual[:100]!r})"
                    )
            except subprocess.TimeoutExpired:
                details.append(f"Test {i+1}: TIMEOUT")
            except Exception as e:
                details.append(f"Test {i+1}: ERROR ({str(e)[:200]})")

        return {
            "correct": passed == total,
            "score": 1.0 if passed == total else passed / total,
            "detail": f"{passed}/{total} passed. " + "; ".join(details),
        }

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM output."""
        # Try <ANSWER>...</ANSWER> block first
        answer_match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
        if answer_match:
            text = answer_match.group(1)

        blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if not blocks:
            blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        if blocks:
            for block in reversed(blocks):
                code = block.strip()
                try:
                    compile(code, "<check>", "exec")
                    return code
                except SyntaxError:
                    continue
            return blocks[-1].strip()

        # Fallback: trained models may output raw Python without markdown fences.
        # Try to extract the longest compilable code region containing definitions.
        lines = text.strip().splitlines()
        best = ""
        for start in range(len(lines)):
            if not re.match(r"^(import |from |def |class |@)", lines[start]):
                continue
            for end in range(len(lines), start, -1):
                candidate = "\n".join(lines[start:end]).strip()
                if not candidate:
                    continue
                try:
                    compile(candidate, "<check>", "exec")
                    if len(candidate) > len(best):
                        best = candidate
                    break
                except SyntaxError:
                    continue
        if best:
            return best

        # Last resort: if the entire text compiles as Python, use it
        stripped = text.strip()
        if stripped:
            try:
                compile(stripped, "<check>", "exec")
                return stripped
            except SyntaxError:
                pass
        return ""

    def _detect_class_and_method(self, code: str, starter_code: str):
        """Detect the class name and method name from starter code or solution code."""
        # Try to find class and method from starter_code first
        class_match = re.search(r"class\s+(\w+)", starter_code or code)
        # Find the first non-__init__ method in the class
        method_match = re.search(
            r"def\s+(\w+)\s*\(self", starter_code or code
        )
        class_name = class_match.group(1) if class_match else None
        method_name = method_match.group(1) if method_match else None
        # Skip __init__ if present, find next method
        if method_name == "__init__":
            methods = re.findall(r"def\s+(\w+)\s*\(self", starter_code or code)
            method_name = next((m for m in methods if m != "__init__"), None)
        return class_name, method_name

    def _run_function_code(self, code: str, test_input: str, starter_code: str,
                           timeout: int = 10) -> str:
        """Execute function/class-based code by wrapping with a driver.

        For LeetCode-style problems where test_input is one or more lines,
        each line being a Python literal representing a function argument.
        """
        class_name, method_name = self._detect_class_and_method(code, starter_code)

        if not class_name or not method_name:
            # Fallback: try running as stdin-based
            return self._run_code(code, test_input, timeout)

        # Build driver that parses each line of input as an argument
        driver = f"""
import sys, json
from typing import List, Optional, Tuple, Dict, Set

{code}

# Parse input: each line is a Python literal argument
_input_lines = sys.stdin.read().strip().split('\\n')
_args = []
for _line in _input_lines:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _args.append(json.loads(_line))
    except json.JSONDecodeError:
        # Try eval for Python literals that json can't parse (e.g. tuples)
        _args.append(eval(_line))

_sol = {class_name}()
_result = _sol.{method_name}(*_args)
print(_result)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(driver)
            tmp = f.name
        try:
            result = subprocess.run(
                ["python", tmp],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[:500])
            return result.stdout
        finally:
            os.unlink(tmp)

    def _run_code(self, code: str, stdin_input: str, timeout: int = 10) -> str:
        """Execute stdin-based code in a subprocess."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp = f.name
        try:
            result = subprocess.run(
                ["python", tmp],
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr[:500])
            return result.stdout
        finally:
            os.unlink(tmp)
