# Autonomous Corkboard — Multi-Agent Task Room

An interactive, visual corkboard dashboard and execution engine for autonomous multi-agent task planning, execution, debate, and verification.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)
![TailwindCSS](https://img.shields.io/badge/Tailwind-CSS-38bdf8.svg)

---

## ✨ Features

- 📌 **Interactive Corkboard UI**: Drag, resize, color-code, and pin tasks as sticky notes on an infinite canvas with dark/light mode.
- 🎭 **Debate & Execution Theater**: Zoom into any task to view real-time agent reasoning, plan formulation, tool executions, and terminal logs via WebSockets.
- 📁 **Task File Browser & Markdown Renderer**:
  - Inspect files created by agents directly in the browser.
  - Built-in `.md` renderer with an instant **Preview (Rendered HTML)** vs **Raw (Syntax Highlighted)** toggle.
  - Download individual workspaces as full `.ZIP` archives.
- 🤖 **Hybrid Multi-Agent Architecture**:
  - **Worker Agent**: Performs research, writes scripts, executes shell commands, and builds artifacts.
  - **Critic Agent**: Reviews outputs, verifies goals against criteria, and requests revisions.
  - **Router**: Dynamically delegates between local LLMs (Ollama / Gemma) and cloud models (Gemini, DeepSeek).
- ⚡ **Real-Time Streaming**: Fast WebSocket updates for thinking streams, subtask progression, and execution badges.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- (Optional) [Ollama](https://ollama.com/) running locally for local LLM inference (e.g., `gemma2:2b`).

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/corkscrew-dashboard.git
cd corkscrew-dashboard

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy the example environment file and configure any optional API keys:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://127.0.0.1:11434` |
| `DEFAULT_MODEL` | Default model for worker | `gemma2:2b` |
| `MAX_ITERATIONS` | Maximum iterations per task | `3` |
| `GEMINI_API_KEY` | Google Gemini API key (optional) | `""` |
| `DEEPSEEK_API_KEY` | DeepSeek API key (optional) | `""` |
| `ROUTER_ENGINE` | Routing decision strategy | `laya_heuristic` |

### 4. Running the Dashboard

Start the FastAPI application:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**.

---

## 📁 Project Structure

```
├── app/
│   ├── server.py         # FastAPI application, REST endpoints, and WebSocket handler
│   ├── graph.py          # LangGraph state machine orchestrating agent workflows
│   ├── agents.py         # Worker and Critic agent logic (Gemini, DeepSeek, Ollama)
│   ├── router.py         # Decision model router for model delegation
│   ├── tools.py          # Execution tools (terminal commands, search, file operations)
│   ├── queue_worker.py   # Async task queue worker processing queued corkboard tasks
│   ├── db.py             # SQLite database layer (aiosqlite)
│   ├── config.py         # System configuration and path management
│   └── static/
│       └── index.html    # Single-page visual corkboard application (Alpine.js + Tailwind)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── .gitignore            # Excludes databases, task runtimes, logs, and venvs
```

---

## 📄 License

MIT License.
