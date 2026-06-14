# Evaluation Handoff Guide — Reasoning Benchmarks

**Machine**: `scai7` (131.179.168.120)  
**User**: `xw27`  
**Working directory**: `/xuanwu-tank/north/xw27/multi/slime/eval/`

---

## 1. Environment Setup

### What is lm-evaluation-harness (lm_eval)?

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is
the standard open-source framework by EleutherAI for evaluating language models
on academic benchmarks. It supports hundreds of tasks (MMLU, GPQA, MATH, etc.),
handles prompt formatting, few-shot sampling, generation, answer extraction, and
metric computation. We use it with the `local-chat-completions` backend, which
sends requests to a local OpenAI-compatible endpoint (our sglang server).

We extend it with **custom task YAMLs** (in `script/slime_tasks/`) that define
our own prompts and parsing logic to match the reward functions used during
training (slime).

### Installing from scratch (if not using existing env)

```bash
# 1. Create a new conda env
conda create -p /path/to/new_env python=3.12 -y
conda activate /path/to/new_env

# 2. Install lm-evaluation-harness
pip install lm-eval==0.4.11

# 3. Install additional dependencies our custom tasks need
pip install datasets transformers torch

# 4. (Optional) Install sglang if you also want to serve models from this env
pip install 'sglang[all]'
```

If you are on `scai7`, the existing `sglang_env` already has everything
installed — just activate it:

```bash
conda activate /xuanwu-tank/north/xw27/envs/sglang_env
```

### How lm_eval works (overview)

```
┌──────────┐    HTTP POST /v1/chat/completions    ┌──────────────┐
│  lm_eval │ ──────────────────────────────────── │ sglang server│
│ harness  │ <─────── model response ──────────── │ (local GPU)  │
└──────────┘                                      └──────────────┘
     │
     │  1. Loads task YAML (e.g. math500_slime.yaml)
     │  2. Formats each example using doc_to_text template
     │  3. Sends to model via local-chat-completions backend
     │  4. Receives model output (with <think>...</think> and answer)
     │  5. Calls process_results (our custom parser in utils.py)
     │  6. Computes exact_match metric, writes results JSON + samples JSONL
```

**Key files**:
- `script/slime_tasks/*.yaml` — task definitions (prompt, generation config)
- `script/slime_tasks/utils.py` — answer extraction and grading functions
- `--include_path script/slime_tasks` flag tells lm_eval where to find them

### Conda environments

| Env | Path | Use for |
|-----|------|---------|
| `sglang_env` | `/xuanwu-tank/north/xw27/envs/sglang_env` | Running `lm_eval` harness |
| `qwen35_sglang` | `/xuanwu-tank/north/xw27/envs/qwen35_sglang` | Serving Qwen3.5-35B-A3B via sglang |

### `sglang_env` (eval harness env)

```
Python          3.12.12
sglang          0.5.10
lm_eval         0.4.11
torch           2.9.1+cu128
transformers    5.3.0
```

### `qwen35_sglang` (sglang serving env)

```
Python          3.12
sglang          0.5.6.post2  (from main @ commit 7d2c11970)
torch           2.9.1+cu128
transformers    5.3.0
huggingface_hub 1.10.1
```

### GPUs

8× NVIDIA H200 (143 GB each). All models use `tp-size 1` (single GPU).

---

## 2. Starting sglang Server

### Qwen3-30B-A3B custom checkpoint (iter 179 or final_search)

```bash
conda activate /xuanwu-tank/north/xw27/envs/sglang_env

CUDA_VISIBLE_DEVICES=<GPU_ID> python -m sglang.launch_server \
  --model-path /xuanwu-tank/north/xw27/model/final_search_hf \
  --served-model-name final_search \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --tool-call-parser qwen25 \
  --host 0.0.0.0 --port 8051
```

### Qwen3-30B-A3B-Base (HuggingFace)

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Base \
  --served-model-name Qwen3-30B-A3B-Base \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --tool-call-parser qwen25 \
  --host 0.0.0.0 --port 8052
```

### Qwen3.5-35B-A3B (official recommended config)

```bash
conda activate /xuanwu-tank/north/xw27/envs/qwen35_sglang

python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b \
  --host 0.0.0.0 --port 8055 --tp-size 1 \
  --mem-fraction-static 0.9 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Port conventions

| Port | Model |
|------|-------|
| 8050 | Qwen3-30B-A3B_base_math (iter 179) |
| 8051 | Custom checkpoints (e.g. final_search) |
| 8052 | Qwen3-30B-A3B-Base |
| 8053 | search_strict_v3_gspo_cold_then_mask |
| 8055 | Qwen3.5-35B-A3B |

### Verify

```bash
curl http://131.179.168.120:<PORT>/v1/models
```

---

## 3. Running Evaluations

### Environment variables (set before every run)

```bash
conda activate /xuanwu-tank/north/xw27/envs/sglang_env
cd /xuanwu-tank/north/xw27/multi/slime/eval

export URL="http://131.179.168.120:<PORT>/v1"
export MODEL="<served-model-name>"
export OPENAI_API_BASE=$URL
export OPENAI_BASE_URL=$URL
export OPENAI_API_KEY=dummy
```

### lm_eval command

```bash
INCLUDE="--include_path /xuanwu-tank/north/xw27/multi/slime/eval/script/slime_tasks"

lm_eval \
  --model local-chat-completions \
  --model_args model=$MODEL,base_url=$URL/chat/completions,num_concurrent=35,max_retries=3,tokenized_requests=False,tokenizer=$MODEL,max_length=32768 \
  --apply_chat_template \
  --batch_size auto \
  --output_path ./results_$MODEL \
  --gen_kwargs temperature=0 \
  --log_samples \
  $INCLUDE \
  --tasks <TASK_NAME>
```

### Configuration per task

All tasks are defined in `script/slime_tasks/`. Parsing logic lives in `script/slime_tasks/utils.py`.

| Parameter | MATH-500 | MMLU | MMLU-Pro | GPQA Diamond | AIME 2024 | AIME 2025 |
|-----------|----------|------|----------|--------------|-----------|-----------|
| **Task name** | `math500_slime` | `mmlu_slime` | `mmlu_pro_slime` | `gpqa_slime` | `aime24_slime` | `aime25_slime` |
| **Dataset** | `HuggingFaceH4/MATH-500` | `cais/mmlu` (all) | `TIGER-Lab/MMLU-Pro` | `Idavidrein/gpqa` (diamond) | `Maxwell-Jia/AIME_2024` | `math-ai/aime25` |
| **Split** | test | test | test | train (no test split) | train | test |
| **Num fewshot** | 0 | 0 | 0 | 0 | 0 | 0 |
| **System prompt** | None | None | None | None | None | None |
| **Temperature** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **do_sample** | false | false | false | false | false | false |
| **max_gen_toks** | 16384 | 16384 | 16384 | 16384 | 32768 | 32768 |
| **Stop tokens** | `<\|im_end\|>`, `<\|endoftext\|>`, `</s>` | same | same | same | same | same |
| **Choices** | N/A (free-form) | 4 (A-D) | 10 (A-J) | 4 (A-D) | N/A (free-form) | N/A (free-form) |
| **Metric** | exact_match (mean) | exact_match (mean) | exact_match (mean) | exact_match (mean) | exact_match (mean) | exact_match (mean) |

#### lm_eval model_args (same for all tasks)

| Arg | Value |
|-----|-------|
| `model` | `local-chat-completions` |
| `base_url` | `http://131.179.168.120:<PORT>/v1/chat/completions` |
| `num_concurrent` | `35` |
| `max_retries` | `3` |
| `tokenized_requests` | `False` |
| `tokenizer` | `<served-model-name>` |
| `max_length` | `32768` |
| `--apply_chat_template` | yes |
| `--batch_size` | `auto` |
| `--log_samples` | yes |

### Run all 4 benchmarks (copy-paste)

```bash
URL="http://131.179.168.120:8051/v1"
MODEL="final_search"
INCLUDE="--include_path /xuanwu-tank/north/xw27/multi/slime/eval/script/slime_tasks"

ARGS="--model local-chat-completions \
  --model_args model=$MODEL,base_url=$URL/chat/completions,num_concurrent=35,max_retries=3,tokenized_requests=False,tokenizer=$MODEL,max_length=32768 \
  --apply_chat_template --batch_size auto \
  --output_path ./results_$MODEL \
  --gen_kwargs temperature=0 \
  --log_samples $INCLUDE"

lm_eval $ARGS --tasks math500_slime
lm_eval $ARGS --tasks mmlu_slime
lm_eval $ARGS --tasks mmlu_pro_slime
lm_eval $ARGS --tasks gpqa_slime
```

---

## 4. System Prompt and Parsing Logic per Task

### 4A. MATH-500 (`math500_slime`)

**User prompt template** (no system prompt):
```
Solve the following problem. Show your reasoning, then put your final answer in \boxed{}.

{problem}
```

**Parsing** (`utils.process_results_math`):
1. `_strip_think(response)` — takes text after the last `</think>` tag
2. `_last_boxed_only_string()` — finds the last `\boxed{...}` in the text
3. `_remove_boxed()` — extracts content inside `\boxed{}`
4. `mathd_normalize_answer()` — normalizes LaTeX:
   - Converts `tfrac`/`dfrac` → `frac`
   - Strips `\left`, `\right`, `^{\circ}`, `\$`, `\%`
   - Fixes `\frac` and `\sqrt` brace formatting
   - Converts `a/b` → `\frac{a}{b}` for integer a,b
   - Converts `0.5` → `\frac{1}{2}`
   - Strips all spaces
5. `grade_answer_mathd()` — compares normalized model answer vs ground truth

**Ground truth**: `doc["answer"]` (LaTeX string, may contain `\boxed{}`)

### 4B. MMLU (`mmlu_slime`)

**User prompt template** (no system prompt):
```
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {choice_A}
B) {choice_B}
C) {choice_C}
D) {choice_D}
```

**Parsing** (`utils.process_results_mmlu`):
1. `_strip_think(response)` — text after `</think>`
2. Regex: `r"(?i)Answer\s*:\s*\$?\s*([A-D])\$?"` — extracts last match
3. Ground truth: `doc["answer"]` is an integer index (0-3), mapped to A-D

### 4C. MMLU-Pro (`mmlu_pro_slime`)

**User prompt template** (no system prompt):
```
Answer the following multiple choice question. Think step by step, then finish your answer with "the answer is (X)" where X is the correct letter choice.

Question:
{question}
Options:
A. {option_A}
B. {option_B}
...
J. {option_J}
```

**Parsing** (`utils.process_results_mmlu_pro`):
1. `_strip_think(response)` — text after `</think>`
2. Regex: `r"(?i)the\s+answer\s+is\s+\(?([A-J])\)?"` — takes last match
3. Fallback: last standalone letter `[A-J]` in post-think text
4. Ground truth: `doc["answer"]` is a letter string (e.g. `"B"`)

### 4D. GPQA Diamond (`gpqa_slime`)

**Preprocessing** (`utils.process_docs_gpqa`):
- Shuffles the 3 incorrect + 1 correct answer randomly
- Records correct answer position as `"(A)"`, `"(B)"`, `"(C)"`, or `"(D)"`
- Cleans answer text: strips `[title]` tags, double spaces

**User prompt template** (no system prompt):
```
Answer the following multiple choice question. Think step by step, then state your answer clearly.

Question: {Question}
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}
```

**Parsing** (`utils.process_results_gpqa`):
1. `_strip_think(response)` — text after `</think>`
2. Sequential regex patterns (first match wins):
   - `r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])"` (case-insensitive)
   - `r"([A-Z])\s*(?:is\s*(?:the)?\s*correct)"`
   - `r"final\s*(?:answer|option)\s*(?:is|:)?\s*([A-Z])"`
3. Fallback: last standalone capital letter `[A-D]`
4. Ground truth: `doc["answer"]` in `"(B)"` format from `process_docs_gpqa`

### 4E. AIME 2024/2025 (`aime24_slime`, `aime25_slime`)

**User prompt template** (same as MATH-500):
```
Solve the following problem. Show your reasoning, then put your final answer in \boxed{}.

{problem}
```

**Parsing**: Same as MATH-500 (`utils.process_results_math`)

**Difference from MATH-500**: `max_gen_toks=32768` (vs 16384) to allow longer reasoning chains.

### Common: `_strip_think()`

All tasks strip thinking blocks. Logic:
```python
if "</think>" in text:
    return text.split("</think>")[-1]
return text
```

---

## 5. Current Results and File Paths

### Scores (temperature=0, zero-shot CoT)

| Model | MATH-500 | MMLU | MMLU-Pro | GPQA-D |
|-------|----------|------|----------|--------|
| **Qwen3-30B-A3B-Base** | 77.00 | 64.24 | 56.77 | 37.88 |
| **Qwen3-30B-A3B_base_math** (iter 179) | 82.00 | 68.87 | 65.16 | 46.97 |
| **search_strict_v3_gspo** | 81.60 | — | 65.22 | 46.97 |

### Result file paths

Each `results_<MODEL>/<MODEL>/` directory contains:
- `results_<timestamp>.json` — aggregated scores, configs, timing
- `samples_<task>_<timestamp>.jsonl` — per-sample predictions (model output, extracted answer, ground truth, correct/incorrect)

```
results_Qwen3-30B-A3B-Base/Qwen3-30B-A3B-Base/
├── results_2026-04-14T23-15-37.660762.json   # mmlu_pro_slime  → 0.5677
├── results_2026-04-14T20-24-18.279271.json   # mmlu_slime      → 0.6424
├── results_2026-04-09T15-48-35.906420.json   # math500_slime   → 0.7700
├── results_2026-04-09T15-42-42.114368.json   # gpqa_slime      → 0.3788
├── samples_mmlu_pro_slime_*.jsonl
├── samples_mmlu_slime_*.jsonl
├── samples_math500_slime_*.jsonl
└── samples_gpqa_slime_*.jsonl

results_Qwen3-30B-A3B_base_math/Qwen3-30B-A3B_base_math/
├── results_2026-04-15T04-46-46.921469.json   # mmlu_pro_slime  → 0.6516
├── results_2026-04-15T02-30-23.224039.json   # mmlu_slime      → 0.6887
├── results_2026-04-10T07-38-15.692472.json   # math500_slime   → 0.8200
├── results_2026-04-10T07-33-26.714708.json   # gpqa_slime      → 0.4697
├── samples_mmlu_pro_slime_*.jsonl
├── samples_mmlu_slime_*.jsonl
├── samples_math500_slime_*.jsonl
└── samples_gpqa_slime_*.jsonl

results_Qwen3-30B-A3B_base_math_search_strict_v3_gspo_cold_then_mask/...
├── results_2026-04-08T05-20-18.891612.json   # mmlu_pro_slime  → 0.6522
├── results_2026-04-08T05-29-23.851725.json   # math500_slime   → 0.8160
└── results_2026-04-08T05-24-15.339619.json   # gpqa_slime      → 0.4697
```

### Model checkpoints

```
/xuanwu-tank/north/xw27/model/
├── Qwen3-30B-A3B_base_math/iter_0000179_hf/   # HF safetensors, iter 179
├── iter_0000499/                                # distcp format, iter 499
└── final_search_hf/                             # HF safetensors, iter 499 (converted)
```

### Reference scripts

```
script/0407.sh          # Primary lm_eval template (temp=0 and temp=0.6)
script/0401.sh          # base_math model full eval
script/0401_base.sh     # Base model full eval
script/0401_search.sh   # search_strict model full eval
script/slime_tasks/     # All task YAML definitions + utils.py
```

### Checkpoint conversion (distcp → HF)

```bash
cd /xuanwu-tank/north/xw27/multi/slime

PYTHONPATH=. python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/to/iter_XXXXXX \
  --output-dir /path/to/output_hf \
  --origin-hf-dir /path/to/Qwen3-30B-A3B-Base-hf/ \
  --vocab-size 151936
```

Note: `tools/convert_torch_dist_to_hf.py` has been patched to catch `transformer_engine`/`slime` module imports via a try/except fallback in `UnpicklerWrapper`.

---

## 6. Notes

- The `litellm` warning about unmapped models (`This model isn't mapped yet`) is **benign** — no effect on accuracy or performance.
- All tasks use **zero-shot CoT** with no system prompt. The chat template is applied by the model's tokenizer via `--apply_chat_template`.
- Temperature `0` is used for deterministic evaluation. Some scripts also run `temperature=0.6` for sampling-based comparison (see `script/0407.sh`).
- Per-sample JSONL files contain the full model response, extracted answer, ground truth, and match result — useful for error analysis.
