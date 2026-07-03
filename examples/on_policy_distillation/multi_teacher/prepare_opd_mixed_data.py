"""Build the mixed Search + Tau prompt dataset for multi-teacher OPD.

The two domains have INCOMPATIBLE prompt conventions:
  * search: --input-key prompt + --apply-chat-template  (sample.prompt = templated text)
  * tau:    --input-key index   + NO chat template       (sample.prompt = task index)

A single training job has only one global --apply-chat-template flag, so we resolve
the conflict OFFLINE: pre-apply the chat template to the search prompts here (using
slime's own Dataset, so it is byte-identical to training-time templating), keep the
tau index verbatim, tag every row with metadata.domain, and emit one merged JSONL.
The run script then loads it with --input-key prompt and NO --apply-chat-template.

Usage:
    python prepare_opd_mixed_data.py \
        --hf /home/ec2-user/MultiStageRL/Qwen3-8B-Base \
        --search-parquet /home/ec2-user/MultiStageRL/nq_hotpotqa_train/train.parquet \
        --tau-jsonl /home/ec2-user/tau-bench/retail_train_tasks.jsonl \
        --out /home/ec2-user/MultiStageRL/opd_mixed/train.jsonl
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from slime.utils.data import Dataset


def _to_jsonable(x):
    """parquet metadata can contain numpy types; coerce to plain JSON."""
    import numpy as np

    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF dir for the tokenizer (e.g. Qwen3-8B-Base)")
    ap.add_argument("--search-parquet", required=True)
    ap.add_argument("--tau-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--search-input-key", default="prompt")
    ap.add_argument("--tau-input-key", default="index")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)

    n_search = n_tau = 0
    with open(args.out, "w") as fout:
        # ---- Search: reproduce --apply-chat-template exactly via slime Dataset ----
        search_ds = Dataset(
            args.search_parquet,
            tokenizer,
            processor=None,
            max_length=None,
            prompt_key=args.search_input_key,
            apply_chat_template=True,  # <-- bake the template into sample.prompt
        )
        for s in search_ds.origin_samples:
            fout.write(
                json.dumps(
                    {
                        "prompt": s.prompt,  # already chat-templated string
                        "metadata": {"domain": "search"},
                    }
                )
                + "\n"
            )
            n_search += 1

        # ---- Tau: keep the task index verbatim; no templating ----
        with open(args.tau_jsonl) as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = _to_jsonable(row.get("metadata") or {})
                meta["domain"] = "tau"
                fout.write(
                    json.dumps(
                        {
                            "prompt": str(row[args.tau_input_key]),  # tau generate does int(sample.prompt)
                            "metadata": meta,
                        }
                    )
                    + "\n"
                )
                n_tau += 1

    print(f"wrote {n_search} search + {n_tau} tau = {n_search + n_tau} rows -> {args.out}")


if __name__ == "__main__":
    main()
