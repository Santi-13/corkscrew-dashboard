import os
import subprocess
from pathlib import Path
from typing import Dict, Any
from duckduckgo_search import DDGS

BLOCKED_PATTERNS = ["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:", "shutdown", "reboot"]

def validate_command(command: str) -> bool:
    cmd_lower = command.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return False
    return True

def run_bash(command: str, cwd: str, timeout: int = 60) -> Dict[str, Any]:
    if not validate_command(command):
        return {
            "success": False,
            "exit_code": -1,
            "output": "Error: Command blocked by safety policy."
        }
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = proc.stdout
        if proc.stderr:
            output += ("\n[STDERR]\n" + proc.stderr if output else proc.stderr)
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "output": output.strip() if output else "[No output]"
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "exit_code": 124,
            "output": f"Error: Command timed out after {timeout} seconds."
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "output": f"Execution error: {str(e)}"
        }

def write_file(filename: str, content: str, cwd: str) -> Dict[str, Any]:
    try:
        target_path = Path(cwd) / filename if not os.path.isabs(filename) else Path(filename)
        # Safety: only allow writing inside /home/r2-d2
        if not str(target_path.resolve()).startswith("/home/r2-d2"):
            return {"success": False, "error": "Cannot write outside user home directory"}
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(target_path), "bytes_written": len(content.encode("utf-8"))}
    except Exception as e:
        return {"success": False, "error": str(e)}

def read_file(filename: str, cwd: str, max_chars: int = 10000) -> Dict[str, Any]:
    try:
        target_path = Path(cwd) / filename if not os.path.isabs(filename) else Path(filename)
        if not target_path.exists():
            return {"success": False, "error": f"File not found: {filename}"}
        content = target_path.read_text(encoding="utf-8")
        truncated = len(content) > max_chars
        return {
            "success": True,
            "path": str(target_path),
            "content": content[:max_chars],
            "truncated": truncated
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_files(directory: str = ".", cwd: str = ".") -> Dict[str, Any]:
    try:
        target_path = Path(cwd) / directory if not os.path.isabs(directory) else Path(directory)
        if not target_path.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}
        entries = []
        for item in sorted(target_path.iterdir()):
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if not item.is_dir() else None
            })
        return {"success": True, "directory": str(target_path), "entries": entries}
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_web(query: str, max_results: int = 3) -> Dict[str, Any]:
    # 1. Try DuckDuckGo
    try:
        results = []
        with DDGS(timeout=10) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title"), "body": r.get("body"), "href": r.get("href")})
        if results:
            return {"success": True, "results": results}
    except Exception:
        pass

    # 2. Fast Wikipedia OpenSearch API fallback
    try:
        import urllib.request
        import urllib.parse
        import json
        clean_query = query.strip('"\'')
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_query)}&limit={max_results}&namespace=0&format=json"
        req = urllib.request.Request(wiki_url, headers={'User-Agent': 'Mozilla/5.0 (Ubuntu; Linux x86_64)'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if len(data) >= 4 and data[1]:
                results = []
                for i in range(len(data[1])):
                    results.append({
                        "title": data[1][i],
                        "body": data[2][i] if i < len(data[2]) and data[2][i] else data[1][i],
                        "href": data[3][i] if i < len(data[3]) else ""
                    })
                if results:
                    return {"success": True, "results": results}
    except Exception:
        pass

    return {"success": False, "error": "Search provider timed out or returned no results", "results": []}

OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a shell command inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash shell command to execute."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file with UTF-8 text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The relative or absolute file path to write to."
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact full text content to write into the file."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file within the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The file path to read."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to read (default: 10000).",
                        "default": 10000
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory).",
                        "default": "."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet for technical documentation, libraries, or solutions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def execute_tool(name: str, args: Dict[str, Any], cwd: str) -> Dict[str, Any]:
    """Execute a structured tool call safely."""
    try:
        if name == "run_bash":
            return run_bash(command=args.get("command", ""), cwd=cwd)
        elif name == "write_file":
            return write_file(filename=args.get("filename", ""), content=args.get("content", ""), cwd=cwd)
        elif name == "read_file":
            return read_file(filename=args.get("filename", ""), cwd=cwd, max_chars=args.get("max_chars", 10000))
        elif name == "list_files":
            return list_files(directory=args.get("directory", "."), cwd=cwd)
        elif name == "search_web":
            return search_web(query=args.get("query", ""))
        else:
            return {"success": False, "error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

