"""
Smart Support Environment
=========================
Implements the OpenEnv Environment interface with three graded task levels:
  - easy   : intent + empathy scoring only
  - medium : adds escalation correctness
  - hard   : adds fraud detection + multi-intent handling
"""
from __future__ import annotations

import os
import random
import sys
import uuid
from typing import Any, Dict, List, Optional

# ─── Ensure project root on path ─────────────────────────────────────────────
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import smart_client

# ─── Task names exposed via /tasks ───────────────────────────────────────────
TASK_NAMES: List[str] = ["easy", "medium", "hard"]


# ─── Scenario banks per difficulty ───────────────────────────────────────────

EASY_SCENARIOS: List[Dict] = [
    {
        "task_type": "easy",
        "customer_message": "I want a refund for my order #12345. The product came damaged.",
        "expected_intent": "refund",
        "expect_escalate": False,
        "expect_fraud": False,
    },
    {
        "task_type": "easy",
        "customer_message": "Where is my order #54321? I placed it 5 days ago.",
        "expected_intent": "track_order",
        "expect_escalate": False,
        "expect_fraud": False,
    },
    {
        "task_type": "easy",
        "customer_message": "I received the wrong item. Can you help me get the right one?",
        "expected_intent": "complaint",
        "expect_escalate": False,
        "expect_fraud": False,
    },
]

MEDIUM_SCENARIOS: List[Dict] = [
    {
        "task_type": "medium",
        "customer_message": "This is completely unacceptable! I've been waiting 3 weeks for my order and no one is helping me!",
        "expected_intent": "delivery_issue",
        "expect_escalate": True,
        "expect_fraud": False,
    },
    {
        "task_type": "medium",
        "customer_message": "Hola, necesito ayuda con mi pedido. No hablo inglés muy bien.",
        "expected_intent": "language_request",
        "expect_escalate": False,
        "expect_fraud": False,
    },
    {
        "task_type": "medium",
        "customer_message": "I have complained 4 times already and NOTHING has been done. I demand to speak to a manager NOW!",
        "expected_intent": "escalation",
        "expect_escalate": True,
        "expect_fraud": False,
    },
]

HARD_SCENARIOS: List[Dict] = [
    {
        "task_type": "hard",
        "customer_message": "I never authorised this £299 charge on my account. This is fraud!",
        "expected_intent": "fraud",
        "expect_escalate": True,
        "expect_fraud": True,
    },
    {
        "task_type": "hard",
        "customer_message": "I want a refund AND I want to report this as fraud. Someone used my account without permission!",
        "expected_intent": "refund",
        "expect_escalate": True,
        "expect_fraud": True,
    },
    {
        "task_type": "hard",
        "customer_message": "There are three unknown transactions on my card and I also never received my order from last month.",
        "expected_intent": "fraud",
        "expect_escalate": True,
        "expect_fraud": True,
    },
]

ALL_SCENARIOS: Dict[str, List[Dict]] = {
    "easy":   EASY_SCENARIOS,
    "medium": MEDIUM_SCENARIOS,
    "hard":   HARD_SCENARIOS,
}


# ─── Graders (separate rubric per difficulty) ─────────────────────────────────

def _grade_easy(action: smart_client.SmartSupportAction, scenario: Dict) -> float:
    """Easy: intent match (0.5) + empathy in response (0.5)."""
    score = 0.0
    if action.intent == scenario["expected_intent"]:
        score += 0.50
    response_text = (action.response or "").lower()
    empathy_kws = ["sorry", "apologize", "help", "assist", "understand", "resolve", "happy to"]
    if any(kw in response_text for kw in empathy_kws):
        score += 0.50
    return round(min(score, 1.0), 4)


def _grade_medium(action: smart_client.SmartSupportAction, scenario: Dict) -> float:
    """Medium: intent (0.4) + empathy (0.3) + escalation (0.3)."""
    score = 0.0
    if action.intent == scenario["expected_intent"]:
        score += 0.40
    response_text = (action.response or "").lower()
    empathy_kws = ["sorry", "apologize", "help", "assist", "understand", "resolve", "happy to"]
    if any(kw in response_text for kw in empathy_kws):
        score += 0.30
    if bool(action.escalate) == scenario["expect_escalate"]:
        score += 0.30
    return round(min(score, 1.0), 4)


def _grade_hard(action: smart_client.SmartSupportAction, scenario: Dict) -> float:
    """Hard: intent (0.30) + empathy (0.25) + escalation (0.20) + fraud (0.25)."""
    score = 0.0
    if action.intent == scenario["expected_intent"]:
        score += 0.30
    response_text = (action.response or "").lower()
    empathy_kws = ["sorry", "apologize", "help", "assist", "understand", "resolve", "happy to"]
    if any(kw in response_text for kw in empathy_kws):
        score += 0.25
    if bool(action.escalate) == scenario["expect_escalate"]:
        score += 0.20
    if bool(action.is_fraud) == scenario["expect_fraud"]:
        score += 0.25
    return round(min(score, 1.0), 4)


GRADERS = {
    "easy":   _grade_easy,
    "medium": _grade_medium,
    "hard":   _grade_hard,
}


def score_action(
    action: smart_client.SmartSupportAction,
    scenario: Dict,
) -> float:
    task_type = scenario.get("task_type", "easy")
    grader = GRADERS.get(task_type, _grade_easy)
    return grader(action, scenario)


# ─── Environment class ────────────────────────────────────────────────────────

class SmartSupportEnvironment(
    Environment[
        smart_client.SmartSupportAction,
        smart_client.SmartSupportObservation,
        State,
    ]
):
    """
    OpenEnv Environment for Smart Customer Support tasks.

    Task levels:
      - easy   : simple refund/tracking, scored on intent + empathy
      - medium : angry/escalation/language, adds escalation scoring
      - hard   : fraud/multi-intent, adds fraud detection scoring
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._episode_id: str = ""
        self._step_count: int = 0
        self._current_scenario: Dict = {}
        self._task_type: str = "easy"

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs: Any,
    ) -> smart_client.SmartSupportObservation:
        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0

        # Allow callers to pin a task type, or pick randomly
        if task_type and task_type in ALL_SCENARIOS:
            self._task_type = task_type
        else:
            self._task_type = random.choice(TASK_NAMES)

        self._current_scenario = random.choice(ALL_SCENARIOS[self._task_type])

        return smart_client.SmartSupportObservation(
            task_type=self._task_type,
            customer_message=self._current_scenario["customer_message"],
            done=False,
            reward=0.0,
            metadata={"episode_id": self._episode_id, "task_type": self._task_type},
        )

    # ------------------------------------------------------------------
    def step(
        self,
        action: smart_client.SmartSupportAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> smart_client.SmartSupportObservation:
        if not self._current_scenario:
            # reset was never called
            return smart_client.SmartSupportObservation(
                task_type="error",
                customer_message="",
                done=True,
                reward=0.0,
                metadata={"error": "reset not called before step"},
            )

        self._step_count += 1
        reward = score_action(action, self._current_scenario)

        feedback = (
            "Good job!" if reward >= 0.70
            else "Partially correct." if reward >= 0.40
            else "Needs improvement."
        )

        return smart_client.SmartSupportObservation(
            task_type=self._task_type,
            customer_message=self._current_scenario["customer_message"],
            done=True,
            reward=reward,
            metadata={
                "episode_id": self._episode_id,
                "step": self._step_count,
                "task_type": self._task_type,
                "expected_intent": self._current_scenario["expected_intent"],
                "agent_intent": action.intent,
                "feedback": feedback,
            },
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = SmartSupportEnvironment()
    for level in TASK_NAMES:
        obs = env.reset(task_type=level, seed=42)
        print(f"\n[{level.upper()}] {obs.customer_message}")
        action = smart_client.SmartSupportAction(
            intent=env._current_scenario["expected_intent"],
            response="Sorry to hear that — I am happy to help you resolve this.",
            escalate=env._current_scenario["expect_escalate"],
            is_fraud=env._current_scenario["expect_fraud"],
        )
        result = env.step(action)
        print(f"  reward={result.reward}  feedback={result.metadata['feedback']}")