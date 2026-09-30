import asyncio
import os
import shutil
import io
import zipfile
import re
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import db
from config import TASKS_DIR, DEFAULT_MODEL, MAX_ITERATIONS, GEMINI_API_KEY, OLLAMA_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_DEFAULT_MODEL
from queue_worker import TaskQueueWorker, active_websockets, broadcast_event

worker_instance = TaskQueueWorker()

async def prewarm_local_model():
    """Ensure Gemma model is prewarmed into RAM and kept alive forever."""
    try:
        import httpx
        model = await db.get_setting("model_name", DEFAULT_MODEL)
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "keep_alive": -1}
            )
        print(f"[Server] Prewarmed local model ({model}) in Ollama with keep_alive: forever.")
    except Exception as e:
        print(f"[Server] Prewarm notice: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    # Pre-warm local Gemma model in Ollama so there's zero cold-start latency
    asyncio.create_task(prewarm_local_model())
    # Start background task processor
    asyncio.create_task(worker_instance.start())
    yield
    worker_instance.is_running = False

app = FastAPI(title="Multi-Agent Task Board", lifespan=lifespan)

class CreateTaskRequest(BaseModel):
    title: str
    description: str

class InterveneRequest(BaseModel):
    instruction: str

class SettingsRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    max_iterations: Optional[int] = None
    model_name: Optional[str] = None
    gemini_model_name: Optional[str] = None
    deepseek_model_name: Optional[str] = None
    gemini_effort_level: Optional[str] = None

class RevertRequest(BaseModel):
    event_id: int

class ForkRequest(BaseModel):
    event_id: int

class AIAssistRequest(BaseModel):
    prompt: str
    mode: str = "interview" # "interview" or "generate"
    history: List[Dict[str, str]] = []
    model: Optional[str] = None

@app.get("/api/tasks")
async def list_tasks():
    return await db.get_all_tasks()

@app.post("/api/tasks")
async def create_task(req: CreateTaskRequest):
    safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', req.title.lower()).strip('_')[:30]
    workspace_name = f"task_{safe_title}" if safe_title else "task"
    workspace = TASKS_DIR / workspace_name
    task_id = await db.create_task(req.title, req.description, str(workspace))
    if not safe_title:
        workspace = TASKS_DIR / f"task_{task_id}"
        async with db.aiosqlite.connect(db.DB_PATH) as conn:
            await conn.execute("UPDATE tasks SET workspace_dir = ? WHERE id = ?", (str(workspace), task_id))
            await conn.commit()
    workspace.mkdir(parents=True, exist_ok=True)
    
    await broadcast_event({"type": "task_created", "task_id": task_id})
    return {"success": True, "task_id": task_id}

@app.get("/api/tasks/{task_id}")
async def get_task_detail(task_id: int):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    events = await db.get_task_events(task_id)
    return {"task": task, "events": events}

@app.delete("/api/tasks/{task_id}")
async def delete_task_endpoint(task_id: int, delete_folder: bool = False):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if delete_folder and task.get("workspace_dir"):
        p = Path(task["workspace_dir"])
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            
    await db.delete_task(task_id)
    await broadcast_event({"type": "task_deleted", "task_id": task_id})
    return {"success": True}

@app.get("/api/tasks/{task_id}/files")
async def list_task_files(task_id: int):
    task = await db.get_task(task_id)
    if not task or not task.get("workspace_dir"):
        raise HTTPException(status_code=404, detail="Task not found")
    
    p = Path(task["workspace_dir"])
    if not p.exists():
        return {"files": [], "path": str(p)}
    
    file_list = []
    for root, _, files in os.walk(p):
        for f in files:
            full = Path(root) / f
            rel = full.relative_to(p)
            try:
                stat = full.stat()
                file_list.append({
                    "name": f,
                    "rel_path": str(rel),
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            except Exception:
                pass
    file_list.sort(key=lambda x: x["rel_path"])
    return {"files": file_list, "path": str(p)}

@app.get("/api/tasks/{task_id}/files/content")
async def get_task_file_content(task_id: int, file: str):
    task = await db.get_task(task_id)
    if not task or not task.get("workspace_dir"):
        raise HTTPException(status_code=404, detail="Task not found")
    
    base = Path(task["workspace_dir"]).resolve()
    target = (base / file).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check for binary file extensions and null bytes
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.pdf', '.zip', '.tar', '.gz',
        '.bz2', '.xz', '.7z', '.exe', '.bin', '.iso', '.so', '.dylib', '.dll', '.pyc',
        '.wasm', '.mp4', '.mp3', '.wav', '.db', '.sqlite', '.sqlite3', '.woff', '.woff2', '.ttf'
    }
    if target.suffix.lower() in binary_extensions:
        return {"is_binary": True, "content": None, "filename": target.name, "size": target.stat().st_size}

    try:
        with open(target, "rb") as f:
            chunk = f.read(2048)
            if b"\x00" in chunk:
                return {"is_binary": True, "content": None, "filename": target.name, "size": target.stat().st_size}
        content = target.read_text(encoding="utf-8", errors="replace")
        return {"is_binary": False, "content": content, "filename": target.name, "size": target.stat().st_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}/download-zip")
async def download_task_zip(task_id: int):
    task = await db.get_task(task_id)
    if not task or not task.get("workspace_dir"):
        raise HTTPException(status_code=404, detail="Task not found")
    
    p = Path(task["workspace_dir"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if p.exists():
            for root, _, files in os.walk(p):
                for f in files:
                    fp = Path(root) / f
                    zf.write(fp, arcname=fp.relative_to(p))
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="task_{task_id}_workspace.zip"'}
    return Response(content=buf.getvalue(), media_type="application/zip", headers=headers)

@app.post("/api/tasks/{task_id}/revert")
async def revert_task(task_id: int, req: RevertRequest):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    await db.revert_task_events(task_id, req.event_id)
    await broadcast_event({"type": "task_updated", "task_id": task_id, "status": "needs_review"})
    return {"success": True}

@app.post("/api/tasks/{task_id}/fork")
async def fork_task(task_id: int, req: ForkRequest):
    parent_task = await db.get_task(task_id)
    if not parent_task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    new_title = f"{parent_task['title']} (Fork #{task_id})"
    new_task_id = await db.create_task(new_title, parent_task["description"], "")
    
    new_workspace = TASKS_DIR / f"task_{new_task_id}"
    parent_workspace = Path(parent_task["workspace_dir"])
    if parent_workspace.exists():
        shutil.copytree(parent_workspace, new_workspace, dirs_exist_ok=True)
    else:
        new_workspace.mkdir(parents=True, exist_ok=True)
        
    async with db.aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute("UPDATE tasks SET workspace_dir = ?, status = 'needs_review' WHERE id = ?", (str(new_workspace), new_task_id))
        events = await db.get_task_events(task_id)
        for ev in events:
            if ev["id"] <= req.event_id:
                await conn.execute("""
                    INSERT INTO task_events (task_id, sender, event_type, content, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (new_task_id, ev["sender"], ev["event_type"], ev["content"], ev["created_at"]))
        await conn.commit()
        
    await broadcast_event({"type": "task_created", "task_id": new_task_id})
    return {"success": True, "new_task_id": new_task_id}

@app.post("/api/tasks/ai-assist")
async def ai_assist_task(req: AIAssistRequest):
    gemini_key = await db.get_setting("gemini_api_key", GEMINI_API_KEY)
    gemini_model = await db.get_setting("gemini_model_name", "gemini-3.7-flash")
    
    if req.mode == "interview":
        system_prompt = (
            "You are an expert technical interviewer helping a user specify a coding or automation task for an autonomous AI agent.\n"
            "The user provided an idea. Ask 1 or 2 sharp, direct clarifying questions to uncover essential requirements, files, or constraints.\n"
            "Keep your response concise, friendly, and formatted as a numbered list of questions."
        )
    else:
        system_prompt = (
            "You are an expert technical task planner.\n"
            "Based on the conversation history and user requirements, produce a high-quality task definition for an autonomous AI agent.\n"
            "You MUST respond ONLY with valid JSON in this exact structure, with no markdown code blocks:\n"
            "{\"title\": \"A concise 4-8 word title\", \"description\": \"Clear, step-by-step technical instructions, files to create, testing steps, and acceptance criteria.\"}"
        )
    
    # Try Gemini if key available
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            chosen = req.model or gemini_model
            if chosen not in ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview"]:
                chosen = "gemini-3.7-flash"
            
            convo_text = ""
            for h in req.history:
                convo_text += f"{h.get('role', 'user').upper()}: {h.get('content', '')}\n"
            convo_text += f"USER: {req.prompt}\n"
            
            prompt = f"{system_prompt}\n\nConversation so far:\n{convo_text}"
            res = client.models.generate_content(model=chosen, contents=prompt)
            reply = res.text.strip()
            if req.mode == "generate":
                clean = reply.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                return {"success": True, "result": parsed}
            else:
                return {"success": True, "reply": reply}
        except Exception:
            pass
    
    # Fallback to local Ollama model
    try:
        import httpx
        local_model = await db.get_setting("model_name", DEFAULT_MODEL)
        convo_text = ""
        for h in req.history:
            convo_text += f"{h.get('role', 'user').upper()}: {h.get('content', '')}\n"
        convo_text += f"USER: {req.prompt}\n"
        prompt = f"{system_prompt}\n\nConversation so far:\n{convo_text}"
        
        async with httpx.AsyncClient(timeout=40.0) as client:
            res = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                "model": local_model,
                "prompt": prompt,
                "stream": False
            })
            if res.status_code == 200:
                reply = res.json().get("response", "").strip()
                if req.mode == "generate":
                    match = re.search(r'\{.*\}', reply, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        return {"success": True, "result": parsed}
                    return {"success": True, "result": {"title": req.prompt[:30], "description": reply}}
                return {"success": True, "reply": reply}
    except Exception:
        pass
        
    if req.mode == "generate":
        return {"success": True, "result": {"title": req.prompt[:30], "description": req.prompt}}
    return {"success": True, "reply": "What specific files, technologies, and testing steps should the agent prioritize?"}

async def _fallback_local_steering(task_id: int, prompt: str, notice: str):
    await broadcast_event({
        "type": "agent_event",
        "task_id": task_id,
        "sender": "gemini",
        "event_type": "status",
        "content": notice
    })
    
    local_model = await db.get_setting("model_name", DEFAULT_MODEL)
    try:
        import ollama
        ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        reply_acc = ""
        last_stream = 0
        import time
        loop = asyncio.get_running_loop()

        def _generate_local():
            nonlocal reply_acc, last_stream
            stream = ollama_client.chat(
                model=local_model,
                messages=[
                    {"role": "system", "content": "You are the Supreme AI Orchestrator overseeing an autonomous agent team on an Ubuntu system."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            for chunk in stream:
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    reply_acc += delta
                    now = time.time()
                    if now - last_stream > 0.05:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_event({
                                "type": "agent_thinking_stream",
                                "task_id": task_id,
                                "sender": "gemini",
                                "content": reply_acc
                            }),
                            loop
                        )
                        last_stream = now
            return reply_acc

        reply = await asyncio.to_thread(_generate_local)
        if not reply:
            reply = "Supreme Orchestrator (Local Model): Reviewing current progress and formulating next steps for the worker agent."

        await db.add_task_event(task_id, "gemini", "orchestration", reply)
        await broadcast_event({
            "type": "agent_event",
            "task_id": task_id,
            "sender": "gemini",
            "event_type": "orchestration",
            "content": reply
        })
    except Exception as local_err:
        err_msg = f"Local Orchestrator error: {local_err}"
        await db.add_task_event(task_id, "gemini", "error", err_msg)
        await broadcast_event({
            "type": "agent_event",
            "task_id": task_id,
            "sender": "gemini",
            "event_type": "error",
            "content": err_msg
        })

async def run_gemini_steering(task_id: int, user_instruction: str):
    task = await db.get_task(task_id)
    if not task:
        return

    # Gather workspace context and file list
    workspace_dir = task.get("workspace_dir") or str(TASKS_DIR / f"task_{task_id}")
    file_tree = []
    try:
        wp = Path(workspace_dir)
        if wp.exists():
            for f in wp.rglob("*"):
                if f.is_file() and not any(part.startswith(".") for part in f.parts):
                    file_tree.append(str(f.relative_to(wp)))
    except Exception:
        pass

    # Gather recent task events (up to last 12 events)
    events = await db.get_task_events(task_id)
    recent_events = events[-12:] if len(events) > 12 else events
    formatted_history = []
    for ev in recent_events:
        formatted_history.append(f"[{ev['sender'].upper()} - {ev['event_type']}]: {ev['content'][:600]}")
    history_str = "\n".join(formatted_history)

    prompt = f"""You are the Supreme AI Orchestrator overseeing an autonomous coding agent team on an Ubuntu system.
The human supervisor has addressed you directly with: "{user_instruction}"

Task Title: {task['title']}
Task Goal: {task['description']}
Workspace Directory: {workspace_dir}
Current Workspace Files: {', '.join(file_tree) if file_tree else 'No files created yet'}

Recent Execution Logs:
{history_str}

Respond decisively with:
1. **Supreme Analysis & Human Answer**: Directly address the supervisor's questions, assessment, or steering requirements.
2. **Supreme Directives for Worker Agent**: Concrete, numbered, step-by-step instructions for the worker agent to execute in the workspace (files to create/edit, commands to run, verification steps).
"""

    gemini_key = await db.get_setting("gemini_api_key", GEMINI_API_KEY)
    if not gemini_key:
        await _fallback_local_steering(task_id, prompt, "Gemini API key not configured. Defaulting to local model Supreme Orchestrator...")
        return

    model_name = await db.get_setting("gemini_model_name", "gemini-3.7-flash")
    effort_level = await db.get_setting("gemini_effort_level", "low")

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_key)

        gen_config = None
        if effort_level and effort_level.lower() != "off":
            budget_map = {"low": 8192, "medium": 24576, "high": 65536}
            budget = budget_map.get(effort_level.lower(), 8192)
            gen_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=budget)
            )

        kwargs = {"model": model_name, "contents": prompt}
        if gen_config:
            kwargs["config"] = gen_config

        loop = asyncio.get_running_loop()

        def _generate():
            import time
            reply_acc = ""
            last_stream = 0
            try:
                for chunk in client.models.generate_content_stream(**kwargs):
                    if chunk.text:
                        reply_acc += chunk.text
                        now = time.time()
                        if now - last_stream > 0.05:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_event({
                                    "type": "agent_thinking_stream",
                                    "task_id": task_id,
                                    "sender": "gemini",
                                    "content": reply_acc
                                }),
                                loop
                            )
                            last_stream = now
                if reply_acc:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_event({
                            "type": "agent_thinking_stream",
                            "task_id": task_id,
                            "sender": "gemini",
                            "content": reply_acc
                        }),
                        loop
                    )
                return reply_acc
            except Exception as e:
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    kwargs["model"] = "gemini-3.7-flash"
                    for chunk in client.models.generate_content_stream(**kwargs):
                        if chunk.text:
                            reply_acc += chunk.text
                    return reply_acc
                raise e

        try:
            reply = await asyncio.wait_for(asyncio.to_thread(_generate), timeout=12.0)
        except asyncio.TimeoutError:
            raise Exception("Gemini request timed out after 12s")

        await db.add_task_event(task_id, "gemini", "orchestration", reply)
        await broadcast_event({
            "type": "agent_event",
            "task_id": task_id,
            "sender": "gemini",
            "event_type": "orchestration",
            "content": reply
        })

    except Exception as e:
        err_msg = str(e)
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            notice = "Gemini quota/credits exhausted. Defaulting to local model Supreme Orchestrator..."
        elif "timed out" in err_msg.lower():
            notice = "Gemini connection timed out. Defaulting to local model Supreme Orchestrator..."
        else:
            notice = f"Gemini offline ({err_msg}). Defaulting to local model Supreme Orchestrator..."
        await _fallback_local_steering(task_id, prompt, notice)

@app.post("/api/tasks/{task_id}/intervene")
async def intervene(task_id: int, req: InterveneRequest):
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    instruction = req.instruction.strip()
    is_gemini = bool(re.search(r'@gemini\b', instruction, re.IGNORECASE) or re.match(r'^\s*gemini\b', instruction, re.IGNORECASE))

    # 1. Record the human intervention event
    await db.add_task_event(task_id, "user", "intervention", instruction)
    await broadcast_event({
        "type": "agent_event",
        "task_id": task_id,
        "sender": "user",
        "event_type": "intervention",
        "content": instruction
    })
    
    # 2. Revert task to active (queued) if in review or completed
    if task["status"] in ("needs_review", "completed"):
        await db.update_task_status(task_id, "queued")
        await broadcast_event({"type": "task_updated", "task_id": task_id, "status": "queued"})
    
    # 3. Handle thinking feedback & routing
    if is_gemini:
        await broadcast_event({
            "type": "agent_event",
            "task_id": task_id,
            "sender": "gemini",
            "event_type": "thinking",
            "content": "Gemini Supreme Orchestrator is analyzing the workspace and preparing steering directives..."
        })
        asyncio.create_task(run_gemini_steering(task_id, instruction))
    else:
        await broadcast_event({
            "type": "agent_event",
            "task_id": task_id,
            "sender": "system",
            "event_type": "thinking",
            "content": "Worker agent is analyzing user guidance and updating execution plan..."
        })

    return {"success": True, "task_id": task_id, "is_gemini": is_gemini}

@app.get("/api/settings")
async def get_settings():
    gemini_key = await db.get_setting("gemini_api_key", GEMINI_API_KEY)
    deepseek_key = await db.get_setting("deepseek_api_key", DEEPSEEK_API_KEY)
    max_iters = await db.get_setting("max_iterations", str(MAX_ITERATIONS))
    model = await db.get_setting("model_name", DEFAULT_MODEL)
    gemini_model = await db.get_setting("gemini_model_name", "gemini-3.7-flash")
    deepseek_model = await db.get_setting("deepseek_model_name", DEEPSEEK_DEFAULT_MODEL)
    gemini_effort = await db.get_setting("gemini_effort_level", "off")
    return {
        "gemini_api_key": gemini_key,
        "deepseek_api_key": deepseek_key,
        "gemini_api_key_configured": bool(gemini_key),
        "deepseek_api_key_configured": bool(deepseek_key),
        "max_iterations": int(max_iters),
        "model_name": model,
        "gemini_model_name": gemini_model,
        "deepseek_model_name": deepseek_model,
        "gemini_effort_level": gemini_effort
    }

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    if req.gemini_api_key is not None and req.gemini_api_key.strip():
        await db.set_setting("gemini_api_key", req.gemini_api_key.strip())
    if req.deepseek_api_key is not None and req.deepseek_api_key.strip():
        await db.set_setting("deepseek_api_key", req.deepseek_api_key.strip())
    if req.max_iterations is not None:
        await db.set_setting("max_iterations", str(req.max_iterations))
    if req.model_name is not None and req.model_name.strip():
        await db.set_setting("model_name", req.model_name.strip())
    if req.gemini_model_name is not None and req.gemini_model_name.strip():
        await db.set_setting("gemini_model_name", req.gemini_model_name.strip())
    if req.deepseek_model_name is not None and req.deepseek_model_name.strip():
        await db.set_setting("deepseek_model_name", req.deepseek_model_name.strip())
    if req.gemini_effort_level is not None and req.gemini_effort_level.strip():
        await db.set_setting("gemini_effort_level", req.gemini_effort_level.strip())
    return {"success": True}

class ValidateKeyRequest(BaseModel):
    api_key: Optional[str] = None

@app.post("/api/settings/validate-deepseek-key")
async def validate_deepseek_key(req: ValidateKeyRequest):
    import httpx
    key = req.api_key.strip() if req.api_key else ""
    if not key:
        key = await db.get_setting("deepseek_api_key", DEEPSEEK_API_KEY)
    if not key:
        return {"valid": False, "error": "No DeepSeek API key entered or configured."}
    
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(
                "https://api.deepseek.com/models",
                headers={"Authorization": f"Bearer {key}"}
            )
            if res.status_code == 200:
                if req.api_key and req.api_key.strip():
                    await db.set_setting("deepseek_api_key", key)
                return {
                    "valid": True,
                    "status": "active",
                    "saved": True,
                    "message": "DeepSeek API key verified and automatically saved (Active)."
                }
            else:
                return {
                    "valid": False,
                    "error": f"HTTP {res.status_code}: {res.text}",
                    "code": res.status_code
                }
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/api/deepseek-models")
async def list_deepseek_models():
    return {
        "models": [
            "deepseek-chat",
            "deepseek-reasoner"
        ]
    }

@app.post("/api/settings/validate-gemini-key")
async def validate_gemini_key(req: ValidateKeyRequest):
    import httpx
    key = req.api_key.strip() if req.api_key else ""
    if not key:
        key = await db.get_setting("gemini_api_key", GEMINI_API_KEY)
    if not key:
        return {"valid": False, "error": "No API key entered or configured."}
    
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            res = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name", "").replace("models/", "") for m in data.get("models", [])]
                if req.api_key and req.api_key.strip():
                    await db.set_setting("gemini_api_key", key)
                return {
                    "valid": True,
                    "status": "active",
                    "saved": True,
                    "models_count": len(models),
                    "message": "API key verified and automatically saved (Active)."
                }
            else:
                try:
                    err_json = res.json().get("error", {})
                    err_msg = err_json.get("message", f"HTTP {res.status_code}")
                except Exception:
                    err_msg = f"HTTP {res.status_code}"
                return {
                    "valid": False,
                    "error": err_msg,
                    "code": res.status_code
                }
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/api/models")
async def list_available_models():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                if models:
                    return {"models": models}
    except Exception:
        pass
    return {"models": [DEFAULT_MODEL, "llama3.2:latest", "gemma2:2b"]}

@app.get("/api/gemini-models")
async def list_gemini_models():
    # Restricted to the user's explicit minimal list
    return {
        "configured": True,
        "models": [
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-3.8-flash",
            "gemini-3.1-pro-preview"
        ]
    }

@app.get("/api/system")
async def get_system_stats():
    import subprocess
    free_out = subprocess.run("free -h", shell=True, capture_output=True, text=True).stdout
    return {"free": free_out}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
    except Exception:
        active_websockets.discard(websocket)

# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
@app.head("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))
