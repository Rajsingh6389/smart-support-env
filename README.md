---
title: Smart Support Env Environment Server
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
---

# Smart Support Env

An AI customer-support reinforcement learning environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv). An agent handles real customer queries and is scored on intent accuracy, empathy, escalation decisions, and fraud detection.

## Environment Variables

Before running, set these in your `.env` file or shell:

```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=hf_...
```

## Quick Start

```bash
# 1. Install deps
pip install "openenv-core[core]>=0.2.1" openai python-dotenv uvicorn

# 2. Start the server
cd server && python app.py

# 3. Run inference (from project root)
python inference.py
```

## Task Levels

| Level | Scoring Rubric |
|---|---|
| `easy` | Intent match (50%) + Empathy in response (50%) |
| `medium` | Intent (40%) + Empathy (30%) + Escalation (30%) |
| `hard` | Intent (30%) + Empathy (25%) + Escalation (20%) + Fraud detection (25%) |

## Action Space (`SmartSupportAction`)

| Field | Type | Description |
|---|---|---|
| `intent` | string | One of: `refund`, `delivery_issue`, `complaint`, `fraud`, `language_request`, `track_order`, `escalation` |
| `response` | string | Empathetic reply to the customer |
| `escalate` | boolean | Whether to escalate to a human agent |
| `is_fraud` | boolean | Whether fraud is detected |
| `language` | string | Detected/requested language code (e.g. `en`, `es`) |
| `order_id` | string | Extracted order ID |

## Observation Space (`SmartSupportObservation`)

| Field | Type | Description |
|---|---|---|
| `task_type` | string | Difficulty: `easy` \| `medium` \| `hard` |
| `customer_message` | string | Raw customer query |
| `reward` | float | Score 0.0-1.0 |
| `done` | boolean | True after first step (single-turn) |
| `metadata` | dict | `expected_intent`, `agent_intent`, `feedback` |

## Inference Output Format

```
[START] task=smart_support env=smart_support_env model=<MODEL>
[STEP] step=1 action=refund reward=0.85 done=true error=null
[END] success=true steps=1 score=0.170 rewards=0.85
```

## Project Structure

```
smart_support_env/
    Dockerfile                         # Container (exposes port 7860)
    inference.py                       # Baseline inference script
    openenv.yaml                       # OpenEnv manifest
    pyproject.toml                     # Project metadata
    smart_client.py                    # Action/Observation models + EnvClient
    server/
        app.py                         # FastAPI server (HTTP + WebSocket)
        smart_support_env_environment.py  # Core environment logic
```

## Docker

```bash
docker build -t smart-support-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  smart-support-env
```
