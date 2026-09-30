import os
import fnmatch
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path("/home/r2-d2/agent-workspace")
# Load .env if present
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
TASKS_DIR = BASE_DIR / "tasks"
LOGS_DIR = BASE_DIR / "logs"

DB_PATH = DATA_DIR / "tasks.db"
SETTINGS_FILE = DATA_DIR / "settings.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemma2:2b")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")

# Decision Model Router Configuration (Laya / Kev / Heuristic)
ROUTER_ENGINE = os.getenv("ROUTER_ENGINE", "laya_heuristic")

# File Viewer Exclude / Ignore Patterns
DEFAULT_FILE_IGNORE_PATTERNS = "__pycache__,*.pyc,venv,.venv,node_modules,.git,.DS_Store,.pytest_cache,*.egg-info"
FILE_VIEWER_IGNORE_PATTERNS = os.getenv("FILE_VIEWER_IGNORE_PATTERNS", DEFAULT_FILE_IGNORE_PATTERNS)

def is_ignored_path(name: str, rel_path: str = "") -> bool:
    """Check if a file or directory name/path matches configured ignore patterns."""
    raw = os.getenv("FILE_VIEWER_IGNORE_PATTERNS", FILE_VIEWER_IGNORE_PATTERNS)
    patterns = [p.strip() for p in raw.split(",") if p.strip()]
    name_lower = name.lower()
    rel_lower = rel_path.lower() if rel_path else ""
    for pat in patterns:
        pat_lower = pat.lower()
        if fnmatch.fnmatch(name_lower, pat_lower):
            return True
        if rel_lower:
            if fnmatch.fnmatch(rel_lower, pat_lower):
                return True
            for part in Path(rel_lower).parts:
                if fnmatch.fnmatch(part, pat_lower):
                    return True
    return False

DATA_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
