"""Multi-teacher on-policy distillation reward module (SGLang teacher mode).

Pure distillation: the task reward is always 0.0; the only learning signal is the
OPD KL penalty applied in slime/backends/megatron_utils/loss.py:

    advantage -= opd_kl_coef * (student_log_prob - teacher_log_prob)   # per token

Each student rollout is scored by the teacher that OWNS its domain (domain-routing,
*not* averaging) — a Math teacher gives meaningless logprobs on a Tau trajectory,
a Search teacher on a Math trajectory, etc. The three teachers are the specialists
from the math2sea / sea2tau / tau2if experiments:

    sample.metadata["domain"] == "math"    ->  Math   teacher  (Qwen3-8B-Base-Math)
    sample.metadata["domain"] == "search"  ->  Search teacher  (Qwen3-8B-Base-Math-SeaSFT-Search)
    sample.metadata["domain"] == "tau"     ->  Tau    teacher  (Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau)

Distilling all three back into ONE student (the end-of-chain
Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau-IF, which has drifted from every earlier
specialty) recovers math + search + tau simultaneously.

Teacher endpoints are read from the environment (the teacher SGLang servers live on
remote nodes / spare GPUs; the script stays config-free):

    OPD_TEACHER_URL_MATH     e.g. http://node-a:13140/generate
    OPD_TEACHER_URL_SEARCH   e.g. http://node-b:13141/generate
    OPD_TEACHER_URL_TAU      e.g. http://node-c:13142/generate

This mirrors slime/rollout/on_policy_distillation.py (the single-teacher example);
the only addition is per-sample teacher selection.
"""

import os

import aiohttp
import torch

from slime.utils.types import Sample

# domain -> teacher /generate endpoint (remote nodes / spare GPUs)
TEACHER_URLS = {
    "math": os.environ["OPD_TEACHER_URL_MATH"],
    "search": os.environ["OPD_TEACHER_URL_SEARCH"],
    "tau": os.environ["OPD_TEACHER_URL_TAU"],
}

# Teacher prefill over a long multi-turn trajectory can be slow; give it room.
_TIMEOUT = aiohttp.ClientTimeout(total=float(os.environ.get("OPD_TEACHER_TIMEOUT", "600")))


def _domain_of(sample: Sample) -> str:
    domain = (sample.metadata or {}).get("domain")
    if domain not in TEACHER_URLS:
        raise ValueError(
            f"sample.metadata['domain'] must be one of {list(TEACHER_URLS)}, got {domain!r}. "
            "Every row of the mixed dataset must be tagged with metadata.domain "
            "(see prepare_opd_mixed_data.py)."
        )
    return domain


async def reward_func(args, sample, **kwargs):
    """Fetch the teacher's token-level logprobs over the student's own trajectory.

    Routed to the teacher that owns this sample's domain. ``max_new_tokens=0`` means
    we only *score* the given ``input_ids`` — no generation happens on the teacher.
    The returned JSON is stored on ``sample.reward`` and consumed in post_process.
    """
    url = TEACHER_URLS[_domain_of(sample)]
    payload = {
        "input_ids": sample.tokens,  # full prompt + multi-turn response (exact student token ids)
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Write teacher logprobs onto each sample; return 0.0 task reward (pure OPD).

    The teacher scores the *entire* trajectory; we keep the response span (the last
    ``response_length`` tokens), which aligns 1:1 with ``sample.loss_mask``. Tokens
    that are masked out (tool/observation turns) still carry a teacher logprob here,
    but contribute nothing because the policy loss multiplies by loss_mask.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
        # sglang: meta_info["input_token_logprobs"] = [[logprob, token_id, ...], ...];
        # the first entry has no logprob (no preceding context), so skip [1:].
        t_log_probs = torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        sample.teacher_log_probs = t_log_probs[-response_length:]

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
