"""
Workspace-Bench Integration for slime Training

This module is the slime custom rollout entry (--custom-generate-function-path
generate_with_ws.generate). It builds a Workspace-Bench environment for the task,
runs a trainable file-editing agent against it, and converts the result to slime's
Sample format. Reward comes from the Bedrock rubric judge inside the env's
terminal step (see ws_bench/envs/base.py + judge.py) — there is NO separate
--reward-function flag, exactly as in the tau-bench migration.
"""

import logging
import os
from typing import Any

from async_env import get_async_env
from ws_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

# Placeholder token id for the degenerate-sample no-op (see res_to_sample). Any
# in-vocab id works: the synthetic sample's loss_mask is ALL ZERO, so these
# tokens never contribute a gradient — they exist only to give get_batch a
# structurally valid (prompt_length >= 1) tensor instead of an empty one.
_ABORT_PLACEHOLDER_TOKEN_ID = 0

# ─────────────────────────────────────────────────────────────────────────────
# Workspace-Bench configuration (AECE / Greenland).
#
# There is NO user simulator (Workspace-Bench is single-shot/agentic), so the
# tau user-sim knobs are gone. Everything is overridable via env so the run
# script is the single source of truth:
#   WS_ENV        environment name (only "workspace")            (default workspace)
#   WS_SPLIT      lite | full (chooses the staged data subset)   (default full)
#   WS_TASK_SPLIT train | dev (which jsonl/index-map slice)       (default train)
#   WS_DATA_DIR   root of the staged dataset on NVMe              (env / default in base.py)
#   WS_MAX_STEPS  max agent turns per trajectory                  (default 30)
# model / model_provider are UNUSED by the SGLang rollout path but RunConfig
# requires them, so we keep placeholder values.
# ─────────────────────────────────────────────────────────────────────────────
WS_CONFIGS = {
    "env": os.environ.get("WS_ENV", "workspace"),
    "agent_strategy": "tool-calling",  # only tool-calling is implemented for training
    "task_split": os.environ.get("WS_TASK_SPLIT", "train"),
    "model_provider": "auto_router",  # Unused, required by RunConfig
    "model": "qwen3.5-4b",  # Unused, required by RunConfig
}
ws_config = RunConfig(**WS_CONFIGS)

WS_DATA_DIR = os.environ.get("WS_DATA_DIR") or None
WS_MAX_STEPS = int(os.environ.get("WS_MAX_STEPS", "30"))


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    Kept VERBATIM from the tau-bench migration: the degenerate / empty /
    truncated-sample guards below are framework-level fixes that protect
    get_batch / GRPO regardless of the benchmark.
    """
    # Map status to slime status
    status_mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    status = status_mapping.get(res.status, Sample.Status.ABORTED)

    # ── Degenerate-sample guard (Option 1: drop from training) ──
    # asolve's env-reset failure path early-returns an EMPTY result
    # (prompt_token_ids=[], response_token_ids=[]), so res.tokens == [] and
    # res.response_length == 0. That sample is ABORTED, but the custom generate
    # path bypasses slime's standard ABORTED filtering, so it reaches the trainer.
    # There, get_batch (megatron_utils/data.py) computes
    #   prompt_length = total_length - response_length
    # and does F.pad(loss_mask, (prompt_length - 1, 1)). With an empty sample
    # prompt_length == 0 -> negative left-pad -> the job-killing
    #   RuntimeError: narrow(): length must be non-negative
    #
    # We cannot physically drop the sample: generate_rollout_async asserts every
    # group has exactly n_samples_per_prompt samples (GRPO needs the full group),
    # so removing one breaks the group. Instead we emit a STRUCTURALLY VALID but
    # ZERO-GRADIENT no-op: a 1-token prompt + 1-token response with loss_mask all
    # zero. prompt_length == 1 keeps F.pad non-negative, and the all-zero
    # loss_mask means this sample contributes nothing to the loss. If the whole
    # task's group degenerates, every sample carries the same reward, so the
    # dynamic_sampling_filter (check_reward_nonzero_std) drops the entire group.
    n_tokens = len(res.tokens) if res.tokens else 0
    resp_len = res.response_length if (getattr(res, "response_length", None) is not None) else 0
    prompt_length = n_tokens - resp_len
    is_degenerate = (n_tokens == 0) or (resp_len <= 0) or (prompt_length <= 0)
    if is_degenerate:
        logger.warning(
            f"res_to_sample: degenerate trajectory for task {task_index} "
            f"(status={res.status}, tokens_len={n_tokens}, response_length={resp_len}, "
            f"prompt_length={prompt_length}); emitting a zero-gradient no-op sample "
            f"(loss_mask all 0) so it is excluded from training without breaking the "
            f"GRPO group."
        )
        noop = Sample(
            index=task_index,
            prompt=res.prompt if res.prompt else str(task_index),
            tokens=[_ABORT_PLACEHOLDER_TOKEN_ID, _ABORT_PLACEHOLDER_TOKEN_ID],
            response="",
            reward=0.0,
            loss_mask=[0],
            status=Sample.Status.ABORTED,
            metadata=res.info,
            rollout_log_probs=[0.0],
            remove_sample=True,
        )
        noop.response_length = 1
        return noop

    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
        rollout_log_probs=res.rollout_log_probs,
    )

    # Truncated trajectories: keep them in the batch with a clearly negative
    # reward instead of dropping. Magnitude (-0.2) is set so truncation is
    # *worse* than a format-bad failure (-0.1); otherwise the model finds it
    # cheaper to keep thinking until the length cap fires than to emit an
    # imperfect tool_call, which directly drives CoT longer.
    if status == Sample.Status.TRUNCATED:
        sample.reward = -0.2

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        if res.loss_mask:
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


def _rollout_uid(sample: Sample, task_index: int) -> str:
    """Stable per-rollout id so the n_samples_per_prompt parallel rollouts of one
    task each get their OWN writable workspace overlay (no cross-contamination).

    Prefer slime's (index, group_index); fall back to a random suffix."""
    gi = getattr(sample, "group_index", None)
    if gi is not None:
        return f"{task_index}-{gi}"
    sid = getattr(sample, "session_id", None)
    if sid:
        return str(sid)
    import secrets

    return f"{task_index}-{secrets.token_hex(6)}"


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for Workspace-Bench.

    This is the main entry point for slime training. It creates a Workspace-Bench
    environment for the task, runs a trainable file-editing agent, and converts
    the result to slime's Sample format.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory
    """
    assert not args.partial_rollout, "Partial rollout is not supported for Workspace-Bench interactions."

    # Extract task index from sample prompt (jsonl rows are {"index": i}).
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # Build the Workspace-Bench environment (async wrapper; the terminal judge
    # call runs in a worker thread, not on the event loop). Each parallel rollout
    # of this task gets its own writable workspace overlay via rollout_uid.
    env = get_async_env(
        env_name=ws_config.env,
        task_split=ws_config.task_split,
        task_index=task_index,
        data_dir=WS_DATA_DIR,
        rollout_uid=_rollout_uid(sample, task_index),
    )

    try:
        # Create trainable agent
        agent = agent_factory(
            tools_info=env.tools_info,
            wiki=env.wiki,
            config=ws_config,
            rollout_args=args,
            sampling_params=sampling_params,
        )

        # Execute agent-environment interaction
        interaction_result = await agent.asolve(
            env, agent.rollout_args, agent.sampling_params, task_index, max_num_steps=WS_MAX_STEPS
        )
    finally:
        # Tear down this rollout's workspace overlay to bound NVMe usage.
        try:
            env.cleanup()
        except Exception as e:
            logger.warning(f"env cleanup failed for task {task_index}: {e}")

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
