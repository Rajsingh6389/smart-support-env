from __future__ import annotations
import os
import sys
import asyncio
from typing import Dict, List, Optional
from pydantic import Field, field_validator, BaseModel
from openai import OpenAI
import uvicorn
from openenv.core.env_server.types import State
from openenv.core.env_server.http_server import create_app

# =========================
#   MODELS (CLEAN)
# =========================
class SmartSupportAction(BaseModel):
    intent: Optional[str] = Field(default=None)
    response: Optional[str] = Field(default=None)
    order_id: Optional[str] = Field(default=None)
    secondary_intent: Optional[str] = Field(default=None)
    escalate: Optional[bool] = Field(default=False)
    status: Optional[str] = Field(default=None)
    is_fraud: Optional[bool] = Field(default=False)
    language: Optional[str] = Field(default=None)

    @field_validator("intent", "secondary_intent", "language", mode="before")
    @classmethod
    def normalize_strings(cls, v):
        if isinstance(v, str): return v.lower().strip()
        return v

class SmartSupportObservation(BaseModel):
    task_type: str = Field(...)
    customer_message: str = Field(...)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Optional[Dict] = Field(default_factory=dict)

# =========================
#   ENVIRONMENT (ASYNC)
# =========================
class SmartSupportEnvironment:
    def __init__(self, **kwargs):
        self.current_task = {"customer_message": "Hello, I need a refund", "intent": "refund"}
        self.step_count = 0

    async def reset_async(self, **kwargs) -> SmartSupportObservation:
        self.step_count = 0
        return SmartSupportObservation(
            task_type="refund_request",
            customer_message=self.current_task["customer_message"]
        )

    async def step_async(self, action: SmartSupportAction) -> SmartSupportObservation:
        self.step_count += 1
        reward = 1.0 if action.intent == self.current_task["intent"] else 0.0
        done = self.step_count >= 1
        return SmartSupportObservation(
            task_type="refund_request",
            customer_message="Thank you",
            reward=reward,
            done=done,
            metadata={"feedback": "Correct!" if reward > 0 else "Wrong intent"}
        )

    async def close_async(self): pass

# =========================
#   APP
# =========================
app = create_app(
    SmartSupportEnvironment,
    SmartSupportAction,
    SmartSupportObservation,
    env_name="smart_support_env"
)

@app.get("/health")
def health(): return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
