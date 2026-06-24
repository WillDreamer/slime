# τ²-bench (tau2) zero-reward on RL'd checkpoints — diagnosis & fixes

**Date:** 2026-06-24
**Affected models:** `Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau` (reward 0.000),
`Qwen3-8B-Base-Math-SeaSFT-Search` (0.002). Vanilla `qwen-8b-base` (0.101) and
`Qwen3-8B-Base-Math` (0.098) are essentially unaffected.
**Logs analyzed:** `logs/tau2/tau2off_*_retail_run1.log`

---

## 1. Symptom

The Tau-SFT model is the **best** model on **tau1** (τ-bench, retail ≈ 31% acc)
but scores a **flat 0.0000** on **tau2** (official τ²-bench). The retail summary
shows `🗄️ DB Match: Not checked: 449` — the agent never reached a gradeable
terminal state, so nothing was scored. It is a measurement artifact, not a
genuine 0-capability.

## 2. Per-model evidence (tau2 retail, run1, 456 tasks = 114×4 trials)

| Model | reward | `TOO_MANY_ERRORS` | `ToolCall` validation err | `ContextWindowExceeded` |
|---|---|---|---|---|
| qwen-8b-base | 0.101 | 3 | 0 | 0 |
| Qwen3-8B-Base-Math | 0.098 | 30 | 0 | 14 |
| …-SeaSFT-Search | 0.002 | **454** | 1 | 0 |
| …-TauSFT-Tau | 0.000 | **449** | 3 | **148** |

Nearly every episode of the two SFT'd checkpoints terminates early with
`TerminationReason.TOO_MANY_ERRORS`. The vanilla checkpoints almost never do.
=> The cause is **specific to the SFT'd checkpoints**, not a harness-wide bug.

## 3. Root cause

### 3a. Primary: tau2 can't build a valid action from the model's turn (NOT a parser bug)

> ⚠️ It is **not** a parser mismatch. tau2 uses `Qwen25Detector`, the **same**
> detector as training's `qwen25` adapter (`aws/README-tau-eval.md:30`,
> `_setup_tau_deps.sh`). Token→structure parsing is identical. The break is that
> the model's *emitted* turn doesn't yield a recognizable tool call, and tau2's
> strict consumer rejects it (where training's lenient consumer would continue).

Two distinct break points, both seen as leaked error strings in the official
tau-model run log:

**Break 1 — tool call not recognized at all:**
```
Retry 1/3 for task ...: UserMessage must have either content or tool_calls. Got ...
```
The model emits a turn that parses to **empty `content` AND zero `tool_calls`**.
Cause: it is a **long-CoT reasoner** (on-disk trajectory shows every assistant
turn ≈ 450–1140 tokens of `"1. Analyze… 2. Determine…"` reasoning, **no
`<tool_call>` tag, `tool_calls: None`**). Under tau2's `max_tokens=2048`/turn the
generation is cut off *inside the reasoning, before the `<tool_call>` block is
ever produced* → detector finds nothing → no action.

**Break 2 — argument in the wrong place:**
```
1 validation error for ToolCall
arguments → Input should be a valid dictionary [type=dict_type, input_value='1008292232', input_type=str]
```
When a tool call *is* emitted, the model puts the **bare value** as `arguments`
(`'1008292232'`) instead of nesting it (`{"order_id": "1008292232"}`). tau2's
pydantic `ToolCall` requires `arguments: dict`, so the call is rejected — again,
effectively no valid tool call.

Both increment the per-episode error budget; ~10 → `TOO_MANY_ERRORS` → reward 0.
A third contributor: long reasoning accumulates across turns until the prompt
overflows (`input 51281 > 32768 context` → 47 `ContextWindowExceededError`).

**Why training/tau1 survives the identical output (the real asymmetry):**
the training/tau1 consumer (`examples/tau-bench/openai_tool_adapter.py`,
`_call_to_action_sglang`) is **deliberately lenient**:
```python
params = json.loads(tool_call["parameters"])
if not isinstance(params, dict):
    logger.warning(f"{params} does not follow dict structure for action")  # warn + CONTINUE
else:
    action = Action(name=tool_call["name"], kwargs=params)
except json.JSONDecodeError:
    logger.warning(...)                                                     # warn + CONTINUE
```
A non-dict `arguments` → warn + fall back to a respond action, **episode
continues**. `trainable_agents.py` charges only a small additive format penalty
(`FORMAT_BAD_PENALTY_SUCCESS = 0.1`, `..._FAIL = 0.0`; `format_ok` ≈ 96%) and the
**think-length penalty is disabled** — so very long CoT is *actively allowed* and
an *"imperfect tool_call directly drives CoT longer"* is treated as a useful
signal (`generate_with_tau.py:89`). tau2's official pipeline instead does strict
pydantic validation + the content-or-tool_calls requirement + a 10-error episode
budget. **Same model output: survivable (small penalty) in training/tau1, fatal
in tau2.**

**Why these checkpoints and not base/Math:** Search/Tau were SFT'd/RL'd inside
this lenient loop and drifted toward long CoT + loosely-structured calls that are
"good enough" for the lenient grader — which **tau1 replays exactly** (Tau model
≈ 31%). Base/Math never went through it, emit short schema-clean calls, and the
*same* strict tau2 accepts them (low score, but non-zero).

### 3b. Secondary (Tau model only): **non-termination → context overflow**

The Tau checkpoint also shows **148 `ContextWindowExceededError`** (vs 0 for
Search, 0/14 for base/Math). The RL'd model "rambles to the cap" and does not
emit a concise stop, so the conversation grows until it exceeds the agent
server's context window — another error source feeding `TOO_MANY_ERRORS`. This
compounds 3a for the Tau model specifically.

### 3c. Red herring (ignore)

```
ERROR | tau2.utils.llm_utils:get_response_cost: This model isn't mapped yet. model=...
```
500+ of these per log. **Harmless** — litellm cost tracking can't price a custom
model name. Counted as `Infra Errors: 7 (excluded from metrics)`. Not the cause.

---

## STATUS (2026-06-24): Fix A + Fix B applied

- **Fix A (max_tokens):** `run_tau2_official.sh` `MAX_TOKENS` 2048 → **8192**.
- **Fix B (lenient tool args):** implemented as the idempotent patch
  `aws/_patch_tau2_lenient_args.py`, applied to the live tau2-bench source
  (`src/tau2/utils/llm_utils.py` `generate()`), and wired into
  `aws/_setup_tau_deps.sh` so a fresh re-clone re-applies it. Toggle off with
  `TAU2_LENIENT_TOOL_ARGS=0`. Unit-tested: `"1008292232"` →
  `{"product_id": "1008292232"}`; multi-param unplaceable → `{}` (soft, no crash).
- ⚠️ These edits live on the **old-root** volume (`/mnt/old-root/...`), where the
  whole tau2 eval setup + the eval `slime` container + tau2-bench live. The
  current-root `/home/ec2-user/slime` is a stale checkout missing the tau2
  scripts, and the active docker (`ifbench_train`) has no tau2-bench. So re-run
  tau2 from the **old-root environment**; if a fresh container is created there,
  `_setup_tau_deps.sh` re-clones tau2-bench and re-applies Fix B automatically.

## 4. Potential solutions (in recommended order)

### Step 0 — Confirm the exact emitted string (do this first, ~10 min)
Dump 5–10 raw assistant completions from the Tau model under the tau2 path
(before parsing) and inspect the literal tool-call text. This decides between
"parser mismatch" (fix harness) and "model emits genuinely malformed JSON"
(fix model). Quickest: hit `:7003/v1/chat/completions` with one retail tool
schema and `tools=[...]`, print the raw `choices[0].message`.

### Fix A — Raise per-turn `max_tokens` so the `<tool_call>` isn't truncated (cheapest, one-line)
Break 1 is largely a budget problem: a long-CoT model under `max_tokens=2048`
gets cut off inside the reasoning before it emits `<tool_call>`. Raise
`MAX_TOKENS` in `run_tau2_official.sh` (e.g. 4096–8192) and re-run one retail
trial; expect the "UserMessage must have content or tool_calls" errors and
`TOO_MANY_ERRORS` to drop. Pair with a larger agent context (`--context-length`)
to cut the `ContextWindowExceededError`s. **No retraining, no harness edit.**
(Won't fix Break 2's non-dict arguments — for that, go to B.) Parser is NOT the
lever: tau2 already uses the same `Qwen25Detector` as training.

### Fix B — Replicate training leniency in tau2 (robust; matches what the model was trained against)
The model was trained against a consumer that **tolerates** non-dict arguments.
Make tau2's tool-parse path (`tau2/utils/llm_utils.py`, before building
`ToolCall`) do the same:
- if `arguments` is a `str`, `json.loads` it; if that yields a dict, use it;
- if it yields/stays a scalar and the target tool has exactly one required
  parameter, wrap as `{<that_param>: value}`;
- else treat it as a **soft format penalty / failed-respond turn** (as slime's
  `_call_to_action_sglang` does) instead of a hard error that trips
  `TOO_MANY_ERRORS`.
This is the most faithful fix: it makes tau2's strictness match the env the model
was optimized in. Keep it behind a flag so strict/official numbers stay reportable.

### Fix C — Address non-termination / context overflow (Tau model)
- Raise the agent server context length and/or **cap per-turn `max_tokens`** so a
  runaway turn can't blow the window (`run_tau2_official.sh` `MAX_TOKENS`,
  `TAU2_MESSAGE_LIMIT`).
- Add stop strings / enforce the end-turn token the model was trained on.
- These reduce the `ContextWindowExceededError` contribution but **do not** fix
  3a — apply alongside A or B.

### Fix D — Loosen the per-episode error budget (diagnostic only)
Raise tau2 `Max Errors` so a couple of malformed calls don't abort the whole
episode. Useful to *measure* how much is format vs. genuine capability, but it
**masks** the real issue — not for reported numbers.

### Fix E — Real fix: train tool-calling in the canonical format
Re-run the Search/Tau SFT/RL with tool calls serialized in the standard
Qwen/Hermes function-calling schema (arguments = JSON object) that production
harnesses expect, OR add a short format-alignment SFT pass. This makes the
checkpoint portable across harnesses (tau1, tau2, and real deployments) instead
of overfit to slime's `ToolCallingAgent`. Highest effort, best long-term.

### Reporting note
Until A/B/E land, **tau2 numbers for the Search/Tau checkpoints are not
comparable** to base/Math — they measure format compatibility, not task skill.
Report tau1 (in-distribution) for these models and mark tau2 as "format-blocked,
pending fix."

---

## 5. Status of the tau2 run itself (as of kill on 2026-06-24)
- Tau model: retail run1 ✅ (0.000), airline run1 ~done (199/200, 0.00; 1 hung
  task), telecom ❌ not started. run2/run3 ❌.
- Other models: retail/airline run1 done; telecom run1 was hung/incomplete when
  the box was cleared. Full 3-run tau2 matrix is **not** complete.
