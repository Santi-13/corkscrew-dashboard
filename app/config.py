import os
from pathlib import Path

BASE_DIR = Path("/home/r2-d2/agent-workspace")
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

DATA_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
