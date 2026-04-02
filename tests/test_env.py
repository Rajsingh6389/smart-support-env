"""
Tests for SmartSupportEnvironment
Run from project root: python -m pytest tests/ -v
"""
import sys
import os

# Ensure root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from server.smart_support_env_environment import (
    SmartSupportEnvironment,
    TASK_NAMES,
    ALL_SCENARIOS,
    score_action,
)
import client


# ─── Helpers ─────────────────────────────────────────────────────────────────

def perfect_action(scenario: dict) -> client.SmartSupportAction:
    """Build an action that matches all expected fields."""
    return client.SmartSupportAction(
        intent=scenario["expected_intent"],
        response="I'm truly sorry to hear that — I'm happy to help you resolve this.",
        escalate=scenario["expect_escalate"],
        is_fraud=scenario["expect_fraud"],
    )


def blank_action() -> client.SmartSupportAction:
    return client.SmartSupportAction(
        intent="complaint",
        response="Ok.",
    )


# ─── Task name tests ─────────────────────────────────────────────────────────

def test_task_names_exist():
    assert set(TASK_NAMES) == {"easy", "medium", "hard"}


def test_all_scenarios_populated():
    for name in TASK_NAMES:
        assert name in ALL_SCENARIOS
        assert len(ALL_SCENARIOS[name]) >= 1


# ─── Reset tests ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", TASK_NAMES)
def test_reset_returns_correct_task_type(level):
    env = SmartSupportEnvironment()
    obs = env.reset(seed=0, task_type=level)
    assert obs.task_type == level
    assert obs.customer_message != ""
    assert obs.done is False
    assert obs.reward == 0.0


# ─── Grader tests ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", TASK_NAMES)
def test_perfect_score_for_level(level):
    env = SmartSupportEnvironment()
    env.reset(seed=42, task_type=level)
    scenario = env._current_scenario
    action = perfect_action(scenario)
    reward = score_action(action, scenario)
    assert reward == 1.0, f"Expected 1.0 for perfect action on {level}, got {reward}"


@pytest.mark.parametrize("level", TASK_NAMES)
def test_partial_score_for_level(level):
    env = SmartSupportEnvironment()
    env.reset(seed=42, task_type=level)
    scenario = env._current_scenario
    action = blank_action()
    reward = score_action(action, scenario)
    assert 0.0 <= reward < 1.0, f"Expected partial score on {level}, got {reward}"


# ─── Step tests ─────────────────────────────────────────────────────────────

def test_step_without_reset_returns_error():
    env = SmartSupportEnvironment()
    obs = env.step(blank_action())
    assert obs.done is True
    assert obs.reward == 0.0
    assert "reset not called" in obs.metadata.get("error", "")


@pytest.mark.parametrize("level", TASK_NAMES)
def test_step_returns_done_and_reward(level):
    env = SmartSupportEnvironment()
    env.reset(seed=1, task_type=level)
    scenario = env._current_scenario
    obs = env.step(perfect_action(scenario))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0
    assert "feedback" in obs.metadata


def test_state_tracks_episode():
    env = SmartSupportEnvironment()
    env.reset(seed=7, task_type="easy")
    assert env.state.step_count == 0
    env.step(blank_action())
    assert env.state.step_count == 1
