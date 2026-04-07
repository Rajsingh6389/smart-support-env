"""
Inference Script - Smart Support Environment
=============================================
- Uses API_BASE_URL + HF_TOKEN for LLM calls via OpenAI client
- Emits structured stdout: [START], [STEP], [END]
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

#     Project root on path                                                      
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import client as env_client

#     Config                                                                    
API_KEY      = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BASE_URL     = os.getenv("BASE_URL", "http://localhost:8000")

TASK_NAME  = "smart_support"
BENCHMARK  = "smart_support_env"
MAX_STEPS  = 5
SUCCESS_SCORE_THRESHOLD = 0.1

#     System prompt                                                             
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI customer support agent.
    Respond to the customer message with a valid JSON object containing:
      intent    - one of: refund, delivery_issue, complaint, fraud,
                  language_request, track_order, escalation
      response  - empathetic reply (must include "sorry" or "help")
      escalate  - true/false
      is_fraud  - true/false
      language  - language code (e.g. "en") or null
      status    - brief status or null
    Return ONLY valid JSON, no markdown fences.
""").strip()


#     Logging                                                                   
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    # Keep action on one line, trim if very long
    action_safe = action.replace("\n", " ")[:100]
    print(
        f"[STEP] step={step} action={action_safe}"
        f" reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


#     LLM   Action                                                              
def get_action(client: OpenAI, step: int,
               customer_message: str) -> env_client.SmartSupportAction:
    user_prompt = f"Step {step}. Customer says: {customer_message}\nReply with JSON only."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text  = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        data = json.loads(text)
        return env_client.SmartSupportAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr)
        return env_client.SmartSupportAction(
            intent="complaint",
            response="Sorry, I'm here to help you with this issue.",
        )


#     Main                                                                      
async def main() -> None:
    client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
    env    = env_client.SmartSupportEnv(base_url=BASE_URL)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # env.reset() returns StepResult; get observation from it
        reset_result = await env.reset()
        obs = reset_result.observation

        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, step, obs.customer_message)

            step_result = await env.step(action)
            reward      = step_result.reward or 0.0
            done        = step_result.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step   = step,
                action = action.intent or "complaint",
                reward = reward,
                done   = done,
                error  = None,
            )

            obs = step_result.observation

            if done:
                break

        score   = min(max(sum(rewards) / MAX_STEPS, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        import traceback
        traceback.print_exc()
        log_step(
            step   = steps_taken + 1,
            action = "error",
            reward = 0.0,
            done   = True,
            error  = str(exc),
        )
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())