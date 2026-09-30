from typing import TypedDict, List, Dict, Any, Callable, Optional
from langgraph.graph import StateGraph, END
from agents import (
    PlannerAgent,
    DeepSeekExecutorAgent,
    LocalGruntAgent,
    CriticAgent,
    evaluate_with_gemini,
    WorkerAgent
)
from router import router, DECISION_LOCAL_GRUNT, DECISION_DEEPSEEK_TOOL

class AgentTaskState(TypedDict):
    task_id: int
    task_prompt: str
    cwd: str
    iteration: int
    max_iterations: int
    subtasks: List[Dict[str, Any]]
    subtask_index: int
    subtask_results: List[Dict[str, Any]]
    actions_log: List[Dict[str, Any]]
    worker_history: List[Dict[str, str]]
    worker_result: Dict[str, Any]
    critic_result: Dict[str, Any]
    gemini_api_key: str
    deepseek_api_key: str
    gemini_model_name: Optional[str]
    local_model_name: Optional[str]
    gemini_effort_level: Optional[str]
    status: str
    result_summary: str
    on_event: Any
    on_stream: Optional[Any]
    on_subtasks: Optional[Any]

def planner_node(state: AgentTaskState) -> AgentTaskState:
    on_event = state["on_event"]
    on_stream = state.get("on_stream")
    on_subtasks = state.get("on_subtasks")
    on_event("system", "status", f"Initiating 3-Tier Multi-LLM Pipeline for: {state['task_prompt'][:60]}...")

    planner = PlannerAgent(
        gemini_key=state.get("gemini_api_key", ""),
        deepseek_key=state.get("deepseek_api_key", ""),
        gemini_model=state.get("gemini_model_name") or "gemini-3.7-flash"
    )
    
    subtasks = planner.plan(
        goal=state["task_prompt"],
        cwd=state["cwd"],
        on_event=on_event,
        on_stream=on_stream
    )

    for i, s in enumerate(subtasks):
        s.setdefault("id", i + 1)
        s.setdefault("status", "pending")
        s.setdefault("assigned_tier", "Pending Router")
        s.setdefault("result_summary", "")

    state["subtasks"] = subtasks
    state["subtask_index"] = 0
    state["subtask_results"] = []
    state["actions_log"] = []

    if on_subtasks:
        on_subtasks(subtasks)

    return state

def executor_node(state: AgentTaskState) -> AgentTaskState:
    on_event = state["on_event"]
    on_stream = state.get("on_stream")
    
    idx = state.get("subtask_index", 0)
    subtasks = state.get("subtasks", [])
    
    if idx >= len(subtasks):
        return state

    subtask = subtasks[idx]
    on_event("system", "status", f"Executing Subtask {idx + 1}/{len(subtasks)}: {subtask.get('title')}")

    # Use System 1 Decision Model Router to select the optimal tier
    decision = router.decide(f"{subtask.get('title')} {subtask.get('instruction')}")
    on_event("system", "routing", f"Decision Model Router -> {decision['tier']} ({decision['reason']})")

    # Update subtask state to running
    subtask["status"] = "running"
    subtask["assigned_tier"] = decision.get("tier", "Tier 2 (DeepSeek API)")
    on_subtasks = state.get("on_subtasks")
    if on_subtasks:
        on_subtasks(subtasks)

    # Route based on decision
    if decision["decision"] == DECISION_LOCAL_GRUNT or not state.get("deepseek_api_key"):
        grunt = LocalGruntAgent(model_name=state.get("local_model_name") or "gemma2:2b")
        res = grunt.execute_grunt(subtask, state["cwd"], on_event, on_stream)
    else:
        executor = DeepSeekExecutorAgent(
            api_key=state.get("deepseek_api_key", "")
        )
        res = executor.execute_subtask(
            subtask=subtask,
            cwd=state["cwd"],
            history=state.get("worker_history", []),
            on_event=on_event,
            on_stream=on_stream
        )

    # Record subtask results and mark completed
    subtask["status"] = "completed" if res.get("success", True) else "failed"
    subtask["result_summary"] = res.get("summary", "")
    if on_subtasks:
        on_subtasks(subtasks)

    state["subtask_results"].append({
        "subtask_id": subtask.get("id", idx + 1),
        "title": subtask.get("title", f"Subtask {idx + 1}"),
        "summary": res.get("summary", ""),
        "success": res.get("success", True)
    })
    
    if res.get("actions"):
        state["actions_log"].extend(res.get("actions"))

    state["subtask_index"] = idx + 1
    return state

def decide_subtask_next(state: AgentTaskState) -> str:
    idx = state.get("subtask_index", 0)
    subtasks = state.get("subtasks", [])
    if idx < len(subtasks):
        return "executor"
    return "auditor"

def auditor_node(state: AgentTaskState) -> AgentTaskState:
    on_event = state["on_event"]
    on_stream = state.get("on_stream")
    on_event("system", "status", "Auditing full deliverable with QA Critic...")

    # Compile deliverables summary
    deliverables_parts = []
    for r in state.get("subtask_results", []):
        deliverables_parts.append(f"### Subtask: {r.get('title')}\n{r.get('summary')}")
    full_deliverables = "\n\n".join(deliverables_parts) if deliverables_parts else "All planned subtasks completed."

    critic = CriticAgent(
        gemini_key=state.get("gemini_api_key", ""),
        deepseek_key=state.get("deepseek_api_key", ""),
        gemini_model=state.get("gemini_model_name") or "gemini-3.7-flash"
    )

    eval_res = critic.evaluate(
        goal=state["task_prompt"],
        deliverables_summary=full_deliverables,
        actions_log=state.get("actions_log", []),
        on_event=on_event,
        on_stream=on_stream
    )

    state["critic_result"] = eval_res
    state["iteration"] += 1

    if eval_res["approved"]:
        state["status"] = "completed"
        state["result_summary"] = full_deliverables + "\n\n[Approved by QA Critic]"
        on_event("system", "completed", "Task approved by QA Auditor.")
    else:
        if state["iteration"] < state["max_iterations"]:
            on_event("system", "status", f"Revising task (Iteration {state['iteration']}/{state['max_iterations']})...")
            # Create a focused revision subtask
            state["subtasks"] = [{
                "id": len(state["subtasks"]) + 1,
                "title": "Address Critic Feedback",
                "instruction": f"Fix the following issues flagged by the auditor:\n{eval_res.get('critique')}",
                "task_type": "tool",
                "expected_deliverable": "Corrected and verified files"
            }]
            state["subtask_index"] = 0
        else:
            state["status"] = "needs_review"
            state["result_summary"] = f"Needs human review:\n{eval_res.get('critique')}"
            on_event("system", "needs_review", "Max iterations reached. Awaiting human input.")

    return state

def decide_audit_next(state: AgentTaskState) -> str:
    if state["status"] == "completed" or state["status"] == "needs_review":
        return END
    return "executor"

def build_agent_graph():
    builder = StateGraph(AgentTaskState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("auditor", auditor_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")

    builder.add_conditional_edges(
        "executor",
        decide_subtask_next,
        {
            "executor": "executor",
            "auditor": "auditor"
        }
    )

    builder.add_conditional_edges(
        "auditor",
        decide_audit_next,
        {
            "executor": "executor",
            END: END
        }
    )

    return builder.compile()
