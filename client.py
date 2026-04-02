from __future__ import annotations
import os
import sys

#   Ensure root is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models import SmartSupportAction, SmartSupportObservation
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State


# Models are now in models.py


class SmartSupportEnv(
    EnvClient[SmartSupportAction, SmartSupportObservation, State]
):

    # =========================
    #   CLEAN PAYLOAD (FIXED)
    # =========================
    def _step_payload(self, action: SmartSupportAction) -> Dict:
        payload = {}

        try:
            #   Pydantic v1/v2 safe
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
    #   SAFE RESULT PARSER
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
    #   STATE PARSER (SAFE)
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