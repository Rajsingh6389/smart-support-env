from __future__ import annotations
from typing import Dict, Optional
from pydantic import Field, field_validator, BaseModel

# =========================
#   ACTION MODEL (CLEAN)
# =========================
class SmartSupportAction(BaseModel):
    """Action for customer support environment"""

    #   Core fields
    intent: Optional[str] = Field(default=None, description="Primary intent")
    response: Optional[str] = Field(default=None, description="Agent response")
    order_id: Optional[str] = Field(default=None, description="Order ID")

    #   Advanced fields
    secondary_intent: Optional[str] = Field(default=None)
    escalate: Optional[bool] = Field(default=False)
    status: Optional[str] = Field(default=None)
    is_fraud: Optional[bool] = Field(default=False)
    language: Optional[str] = Field(default=None)

    # =========================
    #   NORMALIZATION
    # =========================
    @field_validator("intent", "secondary_intent", "language", mode="before")
    @classmethod
    def normalize_strings(cls, v):
        if isinstance(v, str):
            return v.lower().strip()
        return v


# =========================
#   OBSERVATION MODEL (CLEAN)
# =========================
class SmartSupportObservation(BaseModel):
    """Observation returned to agent"""

    task_type: str = Field(..., description="Type of task")
    customer_message: str = Field(..., description="Customer input")

    done: bool = Field(default=False)
    reward: float = Field(default=0.0, ge=0.0, le=1.0)

    metadata: Optional[Dict] = Field(default_factory=dict)

    # =========================
    #   VALIDATION
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
