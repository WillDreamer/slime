# tau2-bench + Search eval — current setup (2026-06-24)

How the τ²-bench (tau2) evaluation of the 4 multi-stage checkpoints is wired,
the serving fixes that were required, and how to run / resume it. Companion to
`TAU2_FORMAT_ISSUE.md` (root-cause analysis of the zero-score bugs).

## 1. Models under test (multi-stage SFT/RL chain)

| Lane name | HF checkpoint | stage |
|---|---|---|
| qwen-8b-base | Qwen/Qwen3-8B-Base | base |
| Qwen3-8B-Base-Math | willhx/Qwen3-8B-Base-Math | + Math SFT |
| Qwen3-8B-Base-Math-SeaSFT-Search | willhx/…-SeaSFT-Search | + Search SFT |
| Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau | willhx/…-TauSFT-Tau | + Tau SFT |

User simulator for all lanes: **GLM-4.7-Flash** (`zai-org/GLM-4.7-Flash`).

## 2. Server topology (8 sglang servers, 1 per H200)

| GPU | Port | Serves | Parsers |
|---|---|---|---|
| 0 | 7000 | qwen-8b-base | `--tool-call-parser qwen` (**no reasoning parser**) |
| 1 | 7001 | Qwen3-8B-Base-Math | `--tool-call-parser qwen` |
| 2 | 7002 | …SeaSFT-Search | `--tool-call-parser qwen` |
| 3 | 7003 | …SeaSFT-Search (was TauSFT-Tau; **repurposed** after tau finished) | `--tool-call-parser qwen` |
| 4 | 7006 | GLM-4.7-Flash (user-sim) | `--tool-call-parser glm47 --reasoning-parser glm45` |
| 5 | 7007 | GLM-4.7-Flash | `glm47 + glm45` |
| 6 | 7008 | GLM-4.7-Flash | `glm47 + glm45` |
| 7 | 7009 | GLM-4.7-Flash | `glm47 + glm45` |

**Agent↔user pairing:** base↔7006, math↔7007, search(runs1-2)↔7008, search(run3)↔7009.

### Why agents run with the reasoning parser OFF (critical)
These checkpoints emit a `<tool_call>` **without** any `<think>…</think>` block.
With `--reasoning-parser qwen3` ON, the whole output (including the `<tool_call>`)
is routed to `reasoning_content`, leaving `content` empty and `tool_calls`
**empty** → every tau2 episode gets no action → 0 reward. Verified by probe.
So agents = tool-call parser only. The **GLM user-sim keeps both parsers**
(`glm47`+`glm45`) so its reasoning is split into `reasoning_content` and the
customer message (`content`) is clean.

## 3. The serving fixes (without these, every model scored ~0)

1. **Stop token, base/math/tau:** `stop:["<|im_end|>"]`. These ckpts don't stop
   after `</tool_call>` — they repeat the call until max_tokens (≈100+ calls/turn)
   → `TOO_MANY_ERRORS`. `<|im_end|>` (which they DO emit after a call) cuts to 1.
2. **Stop token + take-first, search:** the Search-SFT model never emits a turn
   boundary at all (no `<|im_end|>`, no `<|endoftext|>`); it streams continuous
   reasoning + many tool calls and even hallucinates its own tool responses
   (Search-R1 training). Fix = `stop:["</tool_call>","<|im_end|>"]` **plus**
   `TAU2_FIRST_TOOL_CALL=1` (take only the first tool call per turn — see below).
3. **Fix B — lenient tool args** (`TAU2_LENIENT_TOOL_ARGS=1`): coerce a tool call
   whose `arguments` is a bare scalar (`'1008292232'`) into `{param: value}` when
   the tool has one parameter, instead of letting pydantic hard-fail.
4. **max_tokens = 8192** (raised from 2048).

### Patches to tau2-bench (applied by `aws/_setup_tau_deps.sh`, idempotent)
- `aws/_patch_tau2_lenient_args.py` → Fix B in `tau2/utils/llm_utils.py`.
- `aws/_patch_tau2_first_toolcall.py` → take-first (opt-in via `TAU2_FIRST_TOOL_CALL=1`,
  default OFF so only the search lane uses it).
Both are re-applied on every `_setup_tau_deps.sh` run, so a fresh tau2-bench
clone keeps them. Toggle either off via the env var to restore strict behaviour.

## 4. How to run

Servers run inside a container (image `slimerl/slime:latest`, host network, all
GPUs, hf_cache + slime mounted). Launch the 8 servers (see
`aws/autostart-resume.sh` SERVERS array), then:

```bash
# base/math/tau — 4 parallel lanes, 3 runs x {retail,airline,telecom}
RUNS=3 bash eval_scai/run_tau2_4x3_paired.sh

# search runs 1-2 on :7002↔:7008
bash eval_scai/run_tau2_search_only.sh         # (cap RUNS=2 if run3 is on 7003)

# search run3 on the repurposed :7003↔:7009
bash eval_scai/run_tau2_search_run3.sh
```

Per-domain `tau2 run`: `--num-trials 4 --max-steps 200 --max-concurrency 16`.
Results: `tau2-bench/data/simulations/tau2rp_<model>_<domain>_run<i>/` and copied
to `eval_scai/tau2_rp_results/` (persisted). Aggregate with `eval_scai/aggregate_tau.py`.

## 5. Results so far (mean reward over 3 runs)

| Model | retail | airline | telecom |
|---|---|---|---|
| **…TauSFT-Tau** (complete) | **0.36** | **0.09** | **0.49** |
| qwen-8b-base (run1) | 0.11 | 0.44 | ~0.30 |
| Qwen3-8B-Base-Math (run1) | 0.12 | ~0.29 | — |
| …SeaSFT-Search (fixed, in progress) | ~0.08 | — | — |

The tau model went from a flat **0.000** (config artifact) to real, stable
numbers — the whole point of the fixes. Search is genuinely low (search-
specialized) but now measured, not broken.

## 6. Known caveat — infrastructure_error rate
A fraction of sims (~28% on some domains) abort with `infrastructure_error` and
0 messages: litellm intermittently drops `api_base` under concurrency and routes
a call to real OpenAI → `AuthenticationError: dummy` → 4 retries → sim aborts.
These are **excluded** from the mean (reward valid) but reduce N / weaken pass^k.
Mitigation (not yet applied): set `OPENAI_BASE_URL` in env, use the litellm-retry
fork, or lower concurrency.

## 7. Auto-resume
See `aws/searcheval.service` + `aws/autostart-resume.sh`. On a spot stop/start the
service (re)launches all servers and resumes the eval (tau2-bench `--auto-resume`
skips completed `--save-to` dirs). Experiment data lives in the container overlay
on the root volume; model weights + this repo must be present at resume time.
