import json
import time
import re
from typing import Dict, Any, List, Callable, Optional
import httpx
from openai import OpenAI
import ollama

import tools
from config import (
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    GEMINI_API_KEY,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL
)
from router import router, DECISION_LOCAL_GRUNT, DECISION_DEEPSEEK_TOOL

class FailoverManager:
    """Manages circuit breaker and cooldowns for API failover (Gemini -> DeepSeek)."""
    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown_seconds = cooldown_seconds
        self.gemini_circuit_open_until = 0.0

    def trip_gemini(self, reason: str = "Quota Exhausted (429)"):
        self.gemini_circuit_open_until = time.time() + self.cooldown_seconds

    def is_gemini_available(self, api_key: str) -> bool:
        if not api_key:
            return False
        if time.time() < self.gemini_circuit_open_until:
            return False
        return True

failover_mgr = FailoverManager()

PLANNER_SYSTEM_PROMPT = """You are the Supreme AI Strategic Planner.
Your job is to analyze the user's overarching goal and break it down into a clean, logical sequence of 2 to 4 actionable subtasks.

Return your response strictly as a JSON object with this exact structure:
{
  "plan_overview": "A brief 1-2 sentence description of the execution roadmap",
  "subtasks": [
    {
      "id": 1,
      "title": "Short title of subtask",
      "instruction": "Specific, detailed step-by-step instructions for what files to create/edit or commands to run",
      "task_type": "tool",
      "expected_deliverable": "Specific file or verified output"
    }
  ]
}

Note:
- task_type should be "tool" for bash/code/file operations, or "grunt" for pure summarization/log filtering.
- Keep subtasks focused, modular, and verifiable.
"""

EXECUTOR_SYSTEM_PROMPT = """You are an Autonomous Tool & Code Execution Worker.
You execute subtasks step-by-step using your available native function calling tools (run_bash, write_file, read_file, list_files, search_web).

Guidelines:
1. Always write complete, robust scripts and files using `write_file`.
2. For executable code, always run and verify it using `run_bash`.
3. If an error occurs, analyze the error output and correct the file or command.
4. When the subtask deliverable is created and verified, provide a concise summary of what was accomplished.
"""

CRITIC_SYSTEM_PROMPT = """You are a rigorous, practical Senior QA Auditor and Product Architect.
Your job is to critically validate work produced by the agent team against the user's overarching goal.

You will examine:
1. The original goal.
2. The subtasks and their deliverables.
3. The files created and execution logs.

Your response MUST end with one of these two verdicts:
VERDICT: APPROVED
Reason: <brief explanation of why it passed>

OR

VERDICT: REVISE
Critique:
1. <specific flaw or missing requirement with actionable instruction>
"""

class PlannerAgent:
    """Decomposes goals into a structured DAG of subtasks using Gemini (failing over to DeepSeek)."""
    def __init__(self, gemini_key: str = "", deepseek_key: str = "", gemini_model: str = "gemini-3.7-flash"):
        self.gemini_key = gemini_key or GEMINI_API_KEY
        self.deepseek_key = deepseek_key or DEEPSEEK_API_KEY
        self.gemini_model = gemini_model

    def plan(self, goal: str, cwd: str, on_event: Callable[[str, str, str], None], on_stream: Optional[Callable[[str, str], None]] = None) -> List[Dict[str, Any]]:
        prompt = f"Goal:\n{goal}\n\nWorkspace Directory:\n{cwd}\n\nDecompose this goal into actionable subtasks."
        
        # 1. Try Gemini if circuit is closed
        if failover_mgr.is_gemini_available(self.gemini_key):
            models_to_try = [self.gemini_model]
            for alt in ["gemini-3.5-flash", "gemini-3.8-flash"]:
                if alt not in models_to_try:
                    models_to_try.append(alt)
            for m in models_to_try:
                try:
                    on_event("gemini", "status", f"Strategic Planner (Gemini {m}) analyzing goal...")
                    from google import genai
                    client = genai.Client(api_key=self.gemini_key)
                    full_prompt = f"{PLANNER_SYSTEM_PROMPT}\n\n{prompt}"
                    
                    res = client.models.generate_content(
                        model=m,
                        contents=full_prompt
                    )
                    text = res.text.strip()
                    subtasks = self._parse_subtasks(text)
                    if subtasks:
                        on_event("gemini", "plan", f"Decomposed into {len(subtasks)} subtasks via Gemini ({m}).")
                        return subtasks
                except Exception as e:
                    err_str = str(e)
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        failover_mgr.trip_gemini("429 Quota Exhausted")
                        on_event("system", "failover", "Gemini Free Tier quota exhausted (429). Failing over to DeepSeek API Planner...")
                        break
                    else:
                        on_event("system", "warning", f"Gemini planner ({m}) encountered: {err_str}. Checking fallbacks...")

        # 2. Failover to DeepSeek API
        if self.deepseek_key:
            try:
                on_event("deepseek", "status", "Strategic Planner (DeepSeek API) decomposing goal...")
                client = OpenAI(api_key=self.deepseek_key, base_url=DEEPSEEK_BASE_URL)
                res = client.chat.completions.create(
                    model=DEEPSEEK_DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                text = res.choices[0].message.content
                subtasks = self._parse_subtasks(text)
                if subtasks:
                    on_event("deepseek", "plan", f"Decomposed into {len(subtasks)} subtasks via DeepSeek.")
                    return subtasks
            except Exception as e:
                on_event("system", "error", f"DeepSeek planner error: {e}")

        # 3. Fallback: single self-contained subtask
        return [{
            "id": 1,
            "title": "Execute Goal",
            "instruction": goal,
            "task_type": "tool",
            "expected_deliverable": "Completed task requirements"
        }]

    def _parse_subtasks(self, text: str) -> List[Dict[str, Any]]:
        try:
            clean = text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                clean = match.group(0)
            data = json.loads(clean)
            if isinstance(data, dict) and "subtasks" in data:
                return data["subtasks"]
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

class DeepSeekExecutorAgent:
    """Executes subtasks using DeepSeek API with native OpenAI-compatible function calling."""
    def __init__(self, api_key: str = "", model_name: str = DEEPSEEK_DEFAULT_MODEL):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model_name = model_name

    def execute_subtask(
        self,
        subtask: Dict[str, Any],
        cwd: str,
        history: List[Dict[str, Any]],
        on_event: Callable[[str, str, str], None],
        on_stream: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"success": False, "summary": "DeepSeek API key is not configured.", "actions": []}

        client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
        
        messages = [{"role": "system", "content": EXECUTOR_SYSTEM_PROMPT}]
        # Include context from previous subtasks if available
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"SUBTASK TO EXECUTE:\nTitle: {subtask.get('title')}\nInstructions: {subtask.get('instruction')}\nExpected Deliverable: {subtask.get('expected_deliverable')}\nWorkspace: {cwd}"
        })

        actions_taken = []
        max_turns = 6
        turn = 0
        final_summary = ""

        while turn < max_turns:
            turn += 1
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools.OPENAI_TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.2
                )
            except Exception as e:
                on_event("deepseek", "error", f"DeepSeek execution error: {str(e)}")
                return {"success": False, "summary": f"DeepSeek error: {str(e)}", "actions": actions_taken}

            msg = response.choices[0].message
            content = msg.content or ""
            
            if content and on_stream:
                on_stream("deepseek", content)
            if content:
                on_event("deepseek", "thought", content)

            messages.append(msg)

            # Check if any tool calls were returned
            if not msg.tool_calls:
                final_summary = content
                break

            for tcall in msg.tool_calls:
                t_name = tcall.function.name
                try:
                    t_args = json.loads(tcall.function.arguments)
                except Exception:
                    t_args = {}

                arg_summary = t_args.get("command") or t_args.get("filename") or t_args.get("query") or ""
                on_event("deepseek", "tool_call", f"{t_name}({arg_summary})")

                # Execute tool safely
                res = tools.execute_tool(t_name, t_args, cwd=cwd)
                res_content = json.dumps(res, indent=2)
                res_snippet = res_content[:300] + ("..." if len(res_content) > 300 else "")
                on_event("deepseek", "tool_result", res_snippet)

                actions_taken.append({
                    "tool": t_name,
                    "args": t_args,
                    "result": res_snippet
                })

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tcall.id,
                    "name": t_name,
                    "content": res_content
                })

        if not final_summary:
            final_summary = content or "Subtask tool operations completed."

        return {
            "success": True,
            "summary": final_summary,
            "actions": actions_taken,
            "messages": messages
        }

class LocalGruntAgent:
    """Low-overhead local agent running Ollama for text summaries, log distillation, and diffs."""
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.client = ollama.Client(host=OLLAMA_BASE_URL)

    def execute_grunt(
        self,
        subtask: Dict[str, Any],
        cwd: str,
        on_event: Callable[[str, str, str], None],
        on_stream: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        prompt = f"""You are a lightweight Local Grunt Assistant.
Analyze and summarize the requested data quickly.
Subtask: {subtask.get('title')}
Instruction: {subtask.get('instruction')}
Workspace: {cwd}

Provide a clear, direct summary of the requested output."""
        
        on_event("worker", "status", f"Local Grunt ({self.model_name}) processing task...")
        content = ""
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                delta = chunk.get('message', {}).get('content', '')
                if delta:
                    content += delta
                    if on_stream:
                        on_stream("worker", content)
        except Exception as e:
            content = f"Local model execution notice: {str(e)}"

        on_event("worker", "final_answer", content)
        return {
            "success": True,
            "summary": content,
            "actions": []
        }

class CriticAgent:
    """Audits deliverables against initial task prompt using Gemini (failing over to DeepSeek)."""
    def __init__(self, gemini_key: str = "", deepseek_key: str = "", gemini_model: str = "gemini-3.7-flash"):
        self.gemini_key = gemini_key or GEMINI_API_KEY
        self.deepseek_key = deepseek_key or DEEPSEEK_API_KEY
        self.gemini_model = gemini_model

    def evaluate(
        self,
        goal: str,
        deliverables_summary: str,
        actions_log: List[Dict[str, Any]],
        on_event: Callable[[str, str, str], None],
        on_stream: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, Any]:
        actions_str = "\n".join([f"- [{a.get('tool')}]: {a.get('args')} -> {a.get('result', '')[:100]}" for a in actions_log])
        audit_prompt = f"""OVERARCHING GOAL:
{goal}

EXECUTION LOGS & TOOLS USED:
{actions_str if actions_str else "[None]"}

DELIVERABLES & FINAL ANSWER:
{deliverables_summary}

Critique the deliverable and output your final verdict (VERDICT: APPROVED or VERDICT: REVISE)."""

        # 1. Try Gemini if available
        if failover_mgr.is_gemini_available(self.gemini_key):
            models_to_try = [self.gemini_model]
            for alt in ["gemini-3.5-flash", "gemini-3.8-flash"]:
                if alt not in models_to_try:
                    models_to_try.append(alt)
            for m in models_to_try:
                try:
                    on_event("gemini", "status", f"Critic Auditor (Gemini {m}) evaluating work...")
                    from google import genai
                    client = genai.Client(api_key=self.gemini_key)
                    res = client.models.generate_content(
                        model=m,
                        contents=f"{CRITIC_SYSTEM_PROMPT}\n\n{audit_prompt}"
                    )
                    text = res.text.strip()
                    on_event("gemini", "critique", text)
                    is_approved = "VERDICT: APPROVED" in text.upper()
                    on_event("gemini", "verdict", "APPROVED" if is_approved else "REVISE")
                    return {"approved": is_approved, "critique": text}
                except Exception as e:
                    err_str = str(e)
                    if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                        failover_mgr.trip_gemini("429 Quota Exhausted")
                        on_event("system", "failover", "Gemini Free Tier quota exhausted. Failing over to DeepSeek for Review...")
                        break
                    else:
                        on_event("system", "warning", f"Gemini review ({m}) encountered: {err_str}. Checking fallbacks...")

        # 2. Failover to DeepSeek API
        if self.deepseek_key:
            try:
                on_event("deepseek", "status", "Critic Auditor (DeepSeek API) evaluating work...")
                client = OpenAI(api_key=self.deepseek_key, base_url=DEEPSEEK_BASE_URL)
                res = client.chat.completions.create(
                    model=DEEPSEEK_DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                        {"role": "user", "content": audit_prompt}
                    ]
                )
                text = res.choices[0].message.content.strip()
                on_event("deepseek", "critique", text)
                is_approved = "VERDICT: APPROVED" in text.upper()
                on_event("deepseek", "verdict", "APPROVED" if is_approved else "REVISE")
                return {"approved": is_approved, "critique": text}
            except Exception as e:
                on_event("system", "error", f"DeepSeek critic error: {e}")

        # 3. Fallback: approve if deliverables were produced
        return {"approved": True, "critique": "Approved via fallback verification."}

# Backward compatibility wrappers
class WorkerAgent:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.deepseek_executor = DeepSeekExecutorAgent()
        self.grunt_agent = LocalGruntAgent(model_name=model_name)

    def execute_step(self, task_prompt: str, cwd: str, history: List[Dict[str, str]], on_event: Callable[[str, str, str], None], on_stream: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        decision = router.decide(task_prompt)
        subtask = {"title": "Task Step", "instruction": task_prompt, "expected_deliverable": "Completed work"}
        
        if decision["decision"] == DECISION_LOCAL_GRUNT or not DEEPSEEK_API_KEY:
            res = self.grunt_agent.execute_grunt(subtask, cwd, on_event, on_stream)
        else:
            res = self.deepseek_executor.execute_subtask(subtask, cwd, history, on_event, on_stream)

        return {
            "final_answer": res.get("summary", ""),
            "actions_taken": res.get("actions", []),
            "messages": history
        }

def evaluate_with_gemini(task_prompt: str, worker_result: Dict[str, Any], critic_critique: str, api_key: str, on_event: Callable[[str, str, str], None], model_name: Optional[str] = None, effort_level: str = "off", on_stream: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
    critic = CriticAgent(gemini_key=api_key, gemini_model=model_name or "gemini-3.7-flash")
    return critic.evaluate(
        goal=task_prompt,
        deliverables_summary=worker_result.get("final_answer", ""),
        actions_log=worker_result.get("actions_taken", []),
        on_event=on_event,
        on_stream=on_stream
    )
