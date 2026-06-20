import asyncio
import logging
import random
import re

import aiohttp

logger = logging.getLogger(__name__)

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl

_shared_session: aiohttp.ClientSession | None = None


def _get_shared_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(
            limit=64,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=120)
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _shared_session


async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
                raise
            backoff = min(2**attempt, 30) + random.random()
            logger.info(f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
            await asyncio.sleep(backoff)


async def _query_rm_judge(url: str, prompt: str, response: str, max_retries: int = 5) -> float:
    """Query a Skywork-Reward-style sglang /classify endpoint.

    Input: prompt (already chat-template-formatted string from slime) + response text.
    We reconstruct the conversation as [user, assistant] and format with Llama-3.1 template,
    then POST to the sglang /classify endpoint.

    The server returns: [{"embedding": [score]}] for a single input.
    Falls back to 0.0 on failure so training isn't blocked.
    """
    session = _get_shared_session()

    # The prompt is Qwen3 chat-template formatted. Extract raw user content.
    # Format: "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.DOTALL)
    raw_prompt = user_match.group(1) if user_match else prompt

    # Build Llama-3.1 chat template for the Skywork reward model.
    # NOTE: Do NOT include <|begin_of_text|> — sglang adds BOS automatically.
    conv_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{raw_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )

    # Send as a single string (not list) so sglang returns a single dict response.
    payload = {"text": conv_text}
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                # sglang /classify with text=str returns {"embedding": [score], "meta_info": {...}}
                if isinstance(result, dict) and "embedding" in result:
                    return float(result["embedding"][0])
                # Batch response fallback: [{"embedding": [score], ...}]
                elif isinstance(result, list) and result:
                    return float(result[0].get("embedding", [0.0])[0])
                logger.warning(f"rm_judge unexpected response format: {str(result)[:200]}")
                return 0.0
        except Exception as e:
            if attempt + 1 >= max_retries:
                logger.warning(f"rm_judge failed after {max_retries} attempts: {e}. Returning 0.0")
                return 0.0
            backoff = min(2**attempt, 10) + random.random()
            await asyncio.sleep(backoff)
    return 0.0


def _apply_rm_judge_reward(verified_reward: float, rm_score: float, threshold: float) -> float:
    """Combine verified reward with RM judge score.

    Logic:
      - If verified > 0 and rm_score > threshold:  final = verified + 1
      - If verified > 0 and rm_score <= threshold: final = verified - 0.5
      - If verified <= 0:                          final = verified (unchanged)
    """
    if verified_reward > 0:
        if rm_score > threshold:
            return verified_reward + 1.0
        else:
            return verified_reward - 0.5
    return verified_reward


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type in ("ifeval", "multi"):
        from .ifeval import compute_ifeval_reward

        verified = compute_ifeval_reward(response, label, metadata=metadata)
        rm_judge_url = getattr(args, "rm_judge_url", None)
        if rm_judge_url and verified > 0:
            rm_score = await _query_rm_judge(rm_judge_url, sample.prompt, response)
            threshold = getattr(args, "rm_judge_threshold", 0.0)
            return _apply_rm_judge_reward(verified, rm_score, threshold)
        return verified
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
