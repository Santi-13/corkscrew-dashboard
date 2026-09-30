import asyncio
import traceback
from pathlib import Path
from typing import Set, Dict, Any, List, Optional
from fastapi import WebSocket
import db
from config import TASKS_DIR, MAX_ITERATIONS, GEMINI_API_KEY, DEEPSEEK_API_KEY, DEFAULT_MODEL
from graph import build_agent_graph

active_websockets: Set[WebSocket] = set()

async def broadcast_event(event_dict: Dict[str, Any]):
    dead = set()
    for ws in active_websockets:
        try:
            await ws.send_json(event_dict)
        except Exception:
            dead.add(ws)
    active_websockets.difference_update(dead)

class TaskQueueWorker:
    def __init__(self):
        self.is_running = False
        self.current_task_id = None
        self.graph = build_agent_graph()

    async def start(self):
        self.is_running = True
        print("[QueueWorker] Background task processor started.")
        # Recover any orphaned tasks stuck in 'running' upon service startup
        try:
            async with db.aiosqlite.connect(db.DB_PATH) as conn:
                await conn.execute("UPDATE tasks SET status = 'queued' WHERE status = 'running'")
                await conn.commit()
            print("[QueueWorker] Reset any orphaned running tasks to queued.")
        except Exception as e:
            print(f"[QueueWorker] Recovery error: {e}")

        while self.is_running:
            try:
                task = await db.get_next_queued_task()
                if task:
                    await self.process_task(task)
                else:
                    await asyncio.sleep(2.0)
            except Exception as e:
                print(f"[QueueWorker Error]: {e}")
                traceback.print_exc()
                await asyncio.sleep(5.0)

    async def process_task(self, task: Dict[str, Any]):
        task_id = task["id"]
        self.current_task_id = task_id
        task_dir = Path(task["workspace_dir"])
        task_dir.mkdir(parents=True, exist_ok=True)

        await db.update_task_status(task_id, "running")
        await broadcast_event({"type": "task_updated", "task_id": task_id, "status": "running"})

        loop = asyncio.get_running_loop()

        def on_event_sync(sender: str, event_type: str, content: str):
            async def _record():
                await db.add_task_event(task_id, sender, event_type, content)
                await broadcast_event({
                    "type": "agent_event",
                    "task_id": task_id,
                    "sender": sender,
                    "event_type": event_type,
                    "content": content
                })
            asyncio.run_coroutine_threadsafe(_record(), loop)

        def on_stream_sync(sender: str, content: str):
            async def _stream():
                await broadcast_event({
                    "type": "agent_thinking_stream",
                    "task_id": task_id,
                    "sender": sender,
                    "content": content
                })
            asyncio.run_coroutine_threadsafe(_stream(), loop)

        # Retrieve dynamic settings
        max_iters_setting = await db.get_setting("max_iterations", str(MAX_ITERATIONS))
        max_iters = int(max_iters_setting)
        gemini_key = await db.get_setting("gemini_api_key", GEMINI_API_KEY)
        deepseek_key = await db.get_setting("deepseek_api_key", DEEPSEEK_API_KEY)
        local_model = await db.get_setting("model_name", DEFAULT_MODEL)
        gemini_model = await db.get_setting("gemini_model_name", "gemini-3.7-flash")
        gemini_effort = await db.get_setting("gemini_effort_level", "off")

        # Reconstruct dialogue history from existing events so agent incorporates past work,
        # human interventions, and Gemini Supreme Orchestrator directives
        events = await db.get_task_events(task_id)
        worker_history = [{"role": "user", "content": f"TASK REQUIREMENTS:\n{task['description']}"}]

        for ev in events:
            sender = ev.get("sender")
            content = ev.get("content", "")
            etype = ev.get("event_type", "")
            if not content or etype == "thinking":
                continue
            if sender in ("worker", "deepseek") and etype in ("thought", "step", "final_answer", "chat"):
                worker_history.append({"role": "assistant", "content": content})
            elif sender == "user":
                worker_history.append({"role": "user", "content": f"[HUMAN SUPERVISOR GUIDANCE]:\n{content}"})
            elif sender == "gemini":
                worker_history.append({"role": "user", "content": f"[SUPREME ORCHESTRATOR GEMINI DIRECTIVE]:\n{content}"})
            elif sender in ("critic", "deepseek") and etype == "critique":
                worker_history.append({"role": "user", "content": f"[CRITIC FEEDBACK]:\n{content}"})

        # Keep initial task requirement + last 8 events to prevent CPU inference bottleneck
        if len(worker_history) > 9:
            worker_history = [worker_history[0]] + worker_history[-8:]

        current_iter = task.get("iteration", 0)
        target_max_iters = max(max_iters, current_iter + 3)

        def on_subtasks_sync(subtasks: List[Dict[str, Any]]):
            async def _update_subs():
                await db.update_task_subtasks(task_id, subtasks)
                await broadcast_event({
                    "type": "subtasks_updated",
                    "task_id": task_id,
                    "subtasks": subtasks
                })
            asyncio.run_coroutine_threadsafe(_update_subs(), loop)

        initial_state = {
            "task_id": task_id,
            "task_prompt": task["description"],
            "cwd": str(task_dir),
            "iteration": current_iter,
            "max_iterations": target_max_iters,
            "subtasks": [],
            "subtask_index": 0,
            "subtask_results": [],
            "actions_log": [],
            "worker_history": worker_history,
            "worker_result": {},
            "critic_result": {},
            "gemini_api_key": gemini_key,
            "deepseek_api_key": deepseek_key,
            "gemini_model_name": gemini_model,
            "local_model_name": local_model,
            "gemini_effort_level": gemini_effort,
            "status": "running",
            "result_summary": "",
            "on_event": on_event_sync,
            "on_stream": on_stream_sync,
            "on_subtasks": on_subtasks_sync
        }

        try:
            # LangGraph execution in worker thread to prevent blocking asyncio event loop
            final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
            
            final_status = final_state.get("status", "completed")
            summary = final_state.get("result_summary") or final_state.get("worker_result", {}).get("final_answer", "")
            iteration = final_state.get("iteration", 0)

            await db.update_task_status(task_id, final_status, result_summary=summary, iteration=iteration)
            await broadcast_event({
                "type": "task_updated",
                "task_id": task_id,
                "status": final_status,
                "result_summary": summary,
                "iteration": iteration
            })

        except Exception as e:
            err_msg = f"Task execution failed: {str(e)}"
            await db.add_task_event(task_id, "system", "error", err_msg)
            await db.update_task_status(task_id, "failed", result_summary=err_msg)
            await broadcast_event({"type": "task_updated", "task_id": task_id, "status": "failed", "result_summary": err_msg})
        finally:
            self.current_task_id = None
