"""
Decision Model Router (Powered by Official Laya System 1 Decision Model)
Uses convaiinnovations/laya non-autoregressive decision model on CPU
with structured, calibrated choice classification and instant fallback.
"""

from typing import Dict, Any, Optional
import os
import re

DECISION_LOCAL_GRUNT = "LOCAL_GRUNT"
DECISION_DEEPSEEK_TOOL = "DEEPSEEK_TOOL"
DECISION_GEMINI_PLAN = "GEMINI_PLAN"
DECISION_GEMINI_REVIEW = "GEMINI_REVIEW"

TIER_NAMES = {
    DECISION_LOCAL_GRUNT: "Tier 3 (Local Gemma 2B)",
    DECISION_DEEPSEEK_TOOL: "Tier 2 (DeepSeek API)",
    DECISION_GEMINI_PLAN: "Tier 1 (Gemini API)",
    DECISION_GEMINI_REVIEW: "Tier 1 (Gemini API)"
}

ROUTER_QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "Which agent tier should handle this task?",
        "criteria": {
            DECISION_LOCAL_GRUNT: "text summarization, reading logs, file diffs, lightweight extraction",
            DECISION_DEEPSEEK_TOOL: "bash execution, writing code, complex scripts, web search",
            DECISION_GEMINI_PLAN: "high-level planning, architectural roadmap, goal decomposition",
            DECISION_GEMINI_REVIEW: "quality auditing, QA review, verification against requirements"
        }
    }
}

class DecisionModelRouter:
    """
    Native System 1 Decision Model router utilizing Convai Innovations' Laya SDK.
    Runs non-autoregressive classification on CPU with zero generative token hallucinations.
    """
    def __init__(self, model_id: str = "convaiinnovations/laya"):
        self.model_id = model_id
        self._laya_agent = None
        self._init_attempted = False

    def _get_agent(self):
        if not self._init_attempted:
            self._init_attempted = True
            try:
                import laya
                print(f"[Router] Initializing native Laya decision model ({self.model_id}) on CPU...")
                self._laya_agent = laya.load(model_id_or_path=self.model_id, device="cpu")
                print("[Router] Native Laya decision model loaded successfully.")
            except Exception as e:
                print(f"[Router] Laya load notice: {e}")
        return self._laya_agent

    def decide(self, task_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate state + question using native Laya decision model.
        Falls back to instant pattern matching if model is initializing.
        """
        agent = self._get_agent()
        if agent is not None:
            try:
                state = {"task": task_prompt}
                res = agent.predict(state, ROUTER_QUESTIONS)
                tier_ans = res.get("answers", {}).get("tier", {})
                choice = tier_ans.get("choice") or tier_ans.get("value")
                confidence = tier_ans.get("answer_confidence") or tier_ans.get("confidence", 0.95)
                
                if choice in TIER_NAMES:
                    return {
                        "decision": choice,
                        "tier": TIER_NAMES[choice],
                        "confidence": float(confidence),
                        "reason": f"Classified by native Laya Decision Model ({self.model_id}) with {confidence*100:.1f}% confidence.",
                        "engine": "laya_native"
                    }
            except Exception as e:
                print(f"[Router] Laya prediction fallback: {e}")

        # High-speed heuristic fallback
        return self._fallback_decide(task_prompt)

    def _fallback_decide(self, task_prompt: str) -> Dict[str, Any]:
        text = task_prompt.lower()
        
        # QA Review
        if any(re.search(p, text) for p in [r"\b(review|audit|verify|critique|qa|acceptance|evaluate)\b"]):
            return {
                "decision": DECISION_GEMINI_REVIEW,
                "tier": TIER_NAMES[DECISION_GEMINI_REVIEW],
                "confidence": 0.95,
                "reason": "Task requires architectural verification and QA auditing.",
                "engine": "laya_fallback"
            }

        # Strategic Planning
        if any(re.search(p, text) for p in [r"\b(decompose|plan|break down|architecture|roadmap|strategy|orchestrate)\b"]):
            return {
                "decision": DECISION_GEMINI_PLAN,
                "tier": TIER_NAMES[DECISION_GEMINI_PLAN],
                "confidence": 0.92,
                "reason": "Task requires high-level goal decomposition.",
                "engine": "laya_fallback"
            }

        is_grunt = any(re.search(p, text) for p in [
            r"\b(summariz\w*|summary|summaries|distill\w*|condens\w*|filter\w*|clean\w*|reformat\w*|extract\w*|parse\w*|diff)\b",
            r"\bread (?:the )?(?:log|output|file|text|data)\b",
            r"\b(status report|lightweight|inspect log)\b"
        ])
        
        is_tool = any(re.search(p, text) for p in [
            r"\b(write|create|edit|modify|implement|code|script|bash|install|curl|python|pip|git|systemctl)\b",
            r"\b(run (?:bash|command|shell|script|tests))\b",
            r"\b(search|lookup|browse|find online)\b",
            r"\b(fix|debug|refactor|compile)\b",
            r"\b(service|systemd|daemon|docker|server|port)\b"
        ])

        if is_grunt and not (re.search(r"\b(write|create|edit|modify|implement|code|bash|install|curl)\b", text)):
            return {
                "decision": DECISION_LOCAL_GRUNT,
                "tier": TIER_NAMES[DECISION_LOCAL_GRUNT],
                "confidence": 0.94,
                "reason": "Task is text distillation/summarization; assigned to local model to conserve API tokens.",
                "engine": "laya_fallback"
            }

        return {
            "decision": DECISION_DEEPSEEK_TOOL,
            "tier": TIER_NAMES[DECISION_DEEPSEEK_TOOL],
            "confidence": 0.96 if is_tool else 0.85,
            "reason": "Task requires tool execution, bash shell operations, or complex code generation.",
            "engine": "laya_fallback"
        }

router = DecisionModelRouter()
