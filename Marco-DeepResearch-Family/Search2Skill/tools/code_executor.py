import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Dict

# Log file for recording auto-installed packages
_INSTALL_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "logs", "need_install.log"
)


class CodeExecutor:
    """Sandboxed code executor."""

    def __init__(self, timeout: int = 30, helper_config: Dict | None = None,
                 workdir: str | None = None):
        self.timeout = timeout
        self.helper_config = helper_config or {}
        self.workdir = workdir or os.getcwd()

    @classmethod
    def from_config(cls, config) -> "CodeExecutor":
        return cls(
            timeout=config["code_executor"]["timeout"],
            helper_config=config,
            workdir=os.getcwd(),
        )

    def execute(self, code: str) -> Dict:
        """Execute Python code and return {success, stdout, stderr}.

        Auto pip-installs on ModuleNotFoundError/ImportError and retries once.
        """
        code = self._strip_redundant_imports(code)
        result = self._run_code(code)

        # Auto-install on import failure, then retry once
        if not result["success"]:
            module_name = self._extract_missing_module(result["stderr"])
            if module_name:
                install_ok, install_err = self._pip_install(module_name)
                if install_ok:
                    self._log_install(module_name, code, status="installed")
                    result = self._run_code(code)
                else:
                    self._log_install(module_name, code, status="FAILED", error=install_err)

        return result

    def _run_code(self, code: str) -> Dict:
        """Run Python code in a subprocess."""
        helper_preamble = self._build_helper_preamble()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(helper_preamble)
            f.write("\n\n")
            f.write(code)
            tmp_path = f.name

        try:
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                self.workdir
                if not existing_pythonpath else f"{self.workdir}{os.pathsep}{existing_pythonpath}"
            )
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.workdir,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
            }
        finally:
            os.unlink(tmp_path)

    def _extract_missing_module(self, stderr: str) -> str | None:
        """Extract module name from ModuleNotFoundError/ImportError."""
        # ModuleNotFoundError: No module named 'xxx'
        m = re.search(r"ModuleNotFoundError: No module named ['\"]([^'\"\.]+)", stderr)
        if m:
            return m.group(1)
        # ImportError: cannot import name 'xxx' from 'yyy'
        m = re.search(r"ImportError:.*from ['\"]([^'\"\.]+)", stderr)
        if m:
            return m.group(1)
        return None

    def _pip_install(self, module_name: str) -> tuple[bool, str]:
        """Attempt to pip install a module. Returns (success, error_msg)."""
        # Map common module names to pip package names
        pip_name_map = {
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "PIL": "Pillow",
            "rdkit": "rdkit-pypi",
            "openbabel": "openbabel-wheel",
            "Bio": "biopython",
        }
        pip_name = pip_name_map.get(module_name, module_name)
        try:
            result = subprocess.run(
                ["pip", "install", "-q", pip_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip()
        except Exception as e:
            return False, str(e)

    def _log_install(self, module_name: str, code: str, *, status: str = "installed", error: str = ""):
        """Append install record to need_install.log."""
        try:
            os.makedirs(os.path.dirname(_INSTALL_LOG_PATH), exist_ok=True)
            with open(_INSTALL_LOG_PATH, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                code_snippet = code.strip().split("\n")[0][:100]
                line = f"[{timestamp}] {status}: {module_name} | trigger: {code_snippet}"
                if error:
                    line += f" | error: {error}"
                f.write(line + "\n")
        except Exception:
            pass

    def _build_helper_preamble(self) -> str:
        config_json = json.dumps(self.helper_config, ensure_ascii=False)
        return (
            "import warnings as __warnings\n"
            "__warnings.filterwarnings('ignore', message=r\".*doesn.t match a supported version.*\")\n"
            "import json as __json\n"
            "from tools.reference_runtime import ReferenceRuntime as __ReferenceRuntime\n\n"
            f"__REFERENCE_RUNTIME = __ReferenceRuntime(__json.loads(r'''{config_json}'''))\n"
            "fetch_pages = __REFERENCE_RUNTIME.fetch_pages\n"
            "ask_llm_to_extract = __REFERENCE_RUNTIME.ask_llm_to_extract\n"
            "fetch_and_extract_reference = __REFERENCE_RUNTIME.fetch_and_extract_reference\n"
        )

    @staticmethod
    def _strip_redundant_imports(code: str) -> str:
        """Remove `from X import Y` when Y is already defined as a function in the same code."""
        defined = set(re.findall(r'def\s+(\w+)\s*\(', code))
        if not defined:
            return code
        lines = code.split('\n')
        filtered = []
        for line in lines:
            m = re.match(r'\s*from\s+\S+\s+import\s+(.+)', line)
            if m:
                imported = [n.strip().split(' as ')[0].strip() for n in m.group(1).split(',')]
                if all(n in defined for n in imported):
                    continue
            filtered.append(line)
        return '\n'.join(filtered)
