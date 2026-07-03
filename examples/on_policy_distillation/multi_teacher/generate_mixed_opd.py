"""Domain-dispatching rollout for mixed Search + Tau on-policy distillation.

slime allows a single ``--custom-generate-function-path``. To roll out two agentic
domains in one job, this dispatcher routes each prompt to its NATIVE multi-turn
rollout based on ``sample.metadata["domain"]``:

    "search" -> generate_with_search_tools_qwen_sft_no_drift.generate
                (student turns -> local sglang router; search tool -> retriever @ :8000)
    "tau"    -> generate_with_tau.generate
                (student turns -> local sglang router; user turns -> OPENAI_API_BASE)

Both example dirs must be on PYTHONPATH (the run script adds examples/search-r1 and
examples/tau-bench). Each native generate sets sample.tokens / loss_mask /
response_length exactly as in its own RL run, so the OPD teacher-logprob step and the
loss masking downstream are unchanged.
"""

import logging

# Requires examples/search-r1 and examples/tau-bench on PYTHONPATH.
from generate_with_search_tools_qwen_sft_no_drift import generate as _generate_search
from generate_with_tau import generate as _generate_tau

logger = logging.getLogger(__name__)

_DISPATCH = {
    "search": _generate_search,
    "tau": _generate_tau,
}


async def generate(args, sample, sampling_params):
    domain = (sample.metadata or {}).get("domain")
    fn = _DISPATCH.get(domain)
    if fn is None:
        raise ValueError(
            f"sample.metadata['domain'] must be one of {list(_DISPATCH)}, got {domain!r}. "
            "Tag every row in the mixed dataset (see prepare_opd_mixed_data.py)."
        )
    return await fn(args, sample, sampling_params)
