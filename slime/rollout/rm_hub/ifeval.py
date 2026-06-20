"""IFEval-G rule-based reward for training data with rm_type='multi' or 'ifeval'.

Uses the vendored IFEvalG instruction checkers (from allenai/open-instruct)
to score model responses against IFEval-style instruction constraints.

This is distinct from ifbench.py which uses IFBench's 58-instruction held-out
evaluation set. The train dataset (allenai/IF_multi_constraints_upto5) uses
IFEval-G instruction ids that are *not* in IFBench's registry.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .ifeval_g import INSTRUCTION_DICT

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if text:
            normalized.append(text)
    return normalized


def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    if isinstance(raw_kwargs, list):
        processed = [dict(e) if isinstance(e, dict) else {} for e in raw_kwargs]
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs)] * num_instructions
    else:
        processed = [{} for _ in range(num_instructions)]

    # Pad or truncate to match instruction count
    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Strip None values
    return [{k: v for k, v in d.items() if v is not None} for d in processed]


def compute_ifeval_reward(
    response: str,
    label: Any,
    metadata: JsonDict | None = None,
) -> float:
    """Score a model response using IFEvalG instruction checkers.

    Returns the fraction of instructions followed (strict per-instruction
    scoring averaged over all instructions in the row).
    """
    if metadata is None or not response or not response.strip():
        return 0.0

    instruction_ids = _normalize_instruction_ids(
        metadata.get("instruction_id_list") or []
    )
    if not instruction_ids:
        logger.debug("No instruction_id_list in metadata: %s", metadata)
        return 0.0

    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))

    num_followed = 0
    for inst_id, kwargs in zip(instruction_ids, kwargs_list):
        checker_cls = INSTRUCTION_DICT.get(inst_id)
        if checker_cls is None:
            logger.warning("Unknown IFEvalG instruction id: %s — skipping", inst_id)
            continue
        try:
            checker = checker_cls(inst_id)
            checker.build_description(**kwargs)
            if checker.check_following(response):
                num_followed += 1
        except Exception:
            logger.exception("Error evaluating instruction %s", inst_id)

    return num_followed / len(instruction_ids)
