from __future__ import annotations
import os
import sys

# 🔥 Ensure root is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from typing import Dict, Optional
from pydantic import Field, field_validator, BaseModel
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State


# =========================
# 🎯 ACTION MODEL (CLEAN)
# =========================
class SmartSupportAction(BaseModel):
    """Action for customer support environment"""

    # ✅ Core fields
    intent: Optional[str] = Field(default=None, description="Primary intent")
    response: Optional[str] = Field(default=None, description="Agent response")
    order_id: Optional[str] = Field(default=None, description="Order ID")

    # ✅ Advanced fields
    secondary_intent: Optional[str] = Field(default=None)
    escalate: Optional[bool] = Field(default=False)
    status: Optional[str] = Field(default=None)
    is_fraud: Optional[bool] = Field(default=False)
    language: Optional[str] = Field(default=None)

    # =========================
    # 🔥 NORMALIZATION
    # =========================
    @field_validator("intent", "secondary_intent", "language", mode="before")
    @classmethod
    def normalize_strings(cls, v):
        if isinstance(v, str):
            return v.lower().strip()
        return v


# =========================
# 📊 OBSERVATION MODEL (CLEAN)
# =========================
class SmartSupportObservation(BaseModel):
    """Observation returned to agent"""

    task_type: str = Field(..., description="Type of task")
    customer_message: str = Field(..., description="Customer input")

    done: bool = Field(default=False)
    reward: float = Field(default=0.0, ge=0.0, le=1.0)

    metadata: Optional[Dict] = Field(default_factory=dict)

    # =========================
    # 🔥 VALIDATION
    # =========================
    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v):
        allowed = {
            "easy", "medium", "hard",
            "refund_request", "angry_customer",
            "multi_intent", "escalation",
            "order_tracking", "fraud_detection",
            "language_switch"
        }
        if v not in allowed:
            return "unknown"
        return v


class SmartSupportEnv(
    EnvClient[SmartSupportAction, SmartSupportObservation, State]
):

    # =========================
    # 🚀 CLEAN PAYLOAD (FIXED)
    # =========================
    def _step_payload(self, action: SmartSupportAction) -> Dict:
        payload = {}

        try:
            # ✅ Pydantic v1/v2 safe
            data = (
                action.model_dump()
                if hasattr(action, "model_dump")
                else action.dict()
            )

            for field, value in data.items():
                if value is not None and value != "":
                    payload[field] = value

        except Exception as e:
            print("Payload Error:", e)

        return payload

    # =========================
    # 🧪 SAFE RESULT PARSER
    # =========================
    def _parse_result(self, payload: Dict) -> StepResult[SmartSupportObservation]:

        try:
            obs_data = payload.get("observation", payload)

            observation = SmartSupportObservation(
                task_type=obs_data.get("task_type", ""),
                customer_message=obs_data.get("customer_message", ""),
                done=payload.get("done", obs_data.get("done", False)),
                reward=payload.get("reward", obs_data.get("reward", 0.0)),
                metadata=obs_data.get("metadata", {}),
            )

            return StepResult(
                observation=observation,
                reward=observation.reward,
                done=observation.done,
            )

        except Exception as e:
            print("Parse Error:", e)
            return StepResult(
                observation=SmartSupportObservation(
                    task_type="error",
                    customer_message="parse failed",
                    done=True,
                    reward=0.0,
                    metadata={"error": str(e)},
                ),
                reward=0.0,
                done=True,
            )

    # =========================
    # 📊 STATE PARSER (SAFE)
    # =========================
    def _parse_state(self, payload: Dict) -> State:
        try:
            return State(
                episode_id=payload.get("episode_id", ""),
                step_count=payload.get("step_count", 0),
            )
        except Exception as e:
            print("State Parse Error:", e)
            return State(episode_id="error", step_count=0)