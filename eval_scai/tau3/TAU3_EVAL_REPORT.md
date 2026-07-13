# τ³ (tau2-bench v1.0.0) Evaluation — Pipeline & Report

End-to-end documentation for the τ³ evaluation of the four multi-stage
Qwen3-8B checkpoints: the **serving environment**, the **eval pipeline**, and the
**final report** (with the pass^1 denominator convention used in the numbers).

> **τ3 == tau2-bench v1.0.0.** "τ3" is not a separate repo — it is the
> `sierra-research/tau2-bench` v1.0.0 release (same `tau2` CLI), which adds the
> `banking_knowledge` RAG domain, `--task-split-name base|train|test` task
> fixes, and voice. All scripts referenced below live under
> `eval_scai/tau3/`.

---

## 1. Environment

### 1.1 Models under test

Four multi-stage checkpoints (HF repos under `willhx/*`), evaluated as the
**agent**:

| Short name (in results) | HF repo |
|---|---|
| `qwen-8b-base` | `Qwen/Qwen3-8B-Base` |
| `Qwen3-8B-Base-Math` | `willhx/Qwen3-8B-Base-Math` |
| `Qwen3-8B-Base-Math-SeaSFT-Search` | `willhx/Qwen3-8B-Base-Math-SeaSFT-Search` |
| `Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau` | `willhx/Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau` |

The **user simulator** (and the NL-assertion reward judge) is
**GLM-4.7-Flash** (`zai-org/GLM-4.7-Flash`), a ~59 GB MoE model. All five sets
of weights are pre-cached in `hf_cache/` (`HF_HUB_OFFLINE=1`).

### 1.2 Serving stack

The whole stack runs inside the **`slimerl/slime:latest`** docker image
(container name `tau3-srv`, `--gpus all --network host`). To reuse the cached
image with no pull, docker's `data-root` is pointed at
`/mnt/old-data3/var/lib/docker` (via `/etc/docker/daemon.json`, then restart
docker). Mounts: `hf_cache` → `/root/.cache/huggingface`, plus `slime` and
`tau2-bench` under `/home/ec2-user`. Extra pip installs in-container:
`pip install -e tau2-bench` + patch scripts, `rank_bm25` (banking BM25).

### 1.3 GPU topology

**8×H100 80 GB** (agents tp=1; GLM needs tp=2 — the 59 GB MoE will not fit on
one 80 GB card):

| Port | GPU(s) | Role |
|---|---|---|
| 7000 | GPU0 (tp1) | agent — `qwen-8b-base` |
| 7001 | GPU1 (tp1) | agent — `Base-Math` |
| 7002 | GPU2 (tp1) | agent — `Base-Math-SeaSFT-Search` |
| 7003 | GPU3 (tp1) | agent — `…-TauSFT-Tau` |
| 7006 | GPU4-5 (tp2) | GLM-4.7-Flash user-sim (pairs base/math lanes) |
| 7007 | GPU6-7 (tp2) | GLM-4.7-Flash user-sim (pairs search/tau lanes) |

(On the original 8×H200 snapshot GLM fit tp=1 at 7006-7009; the H100 layout
above is the current one. See `serve_tau3_fleet.sh`.)

### 1.4 Critical serving fixes (without these, score ≈ 0)

- Agents served with tool-call parser **`qwen`** and **no reasoning parser**.
- Stop token `<|im_end|>`; the search lane additionally stops on
  `</tool_call>` with `TAU2_FIRST_TOOL_CALL=1`.
- `TAU2_LENIENT_TOOL_ARGS=1`; agent `max_tokens=8192`.
- GLM user-sim uses `glm47` + `glm45` parsers; `GLM_CTX=65536` (raised from
  32768 so long conversations don't overflow the user side).
- **Local reward judge:** `tau2/config.py` hardcoded the NL-assertion judge to
  `gpt-4.1` (and the env-interface / user-sim to OpenAI/Anthropic), which fails
  offline. `slime/aws/_patch_tau2_local_judge.py` reroutes them to the local
  GLM (`TAU2_JUDGE_MODEL` / `TAU2_JUDGE_API_BASE`, exported per-lane by
  `run_tau3.sh`). For offline knowledge eval use `--retrieval-config bm25`
  (no API key / sandbox).

Details in `eval_scai/TAU2_SEARCH_SETUP.md`.

### 1.5 Known context-length limitation (affects the report — see §3.4)

The Qwen3-8B **agents are natively capped at 32768 tokens** (no rope_scaling /
YaRN enabled — a deliberate choice, YaRN was *not* turned on). The agent is the
binding overflow constraint: on the longest retail/telecom tasks the agent
context overflows and the simulation ends as `infrastructure_error`. Raising
`GLM_CTX` alone cannot help. Result: a fraction of the longest retail/telecom
sims never complete and are not scored. airline and mock tasks are short and
are essentially unaffected.

---

## 2. Eval pipeline

### 2.1 Domains and shape

Default `DOMAINS` in `run_tau3.sh` is **`retail airline telecom mock`**.
`banking_knowledge` was **dropped (2026-07-04)** — the agentic-RAG capability
ceiling is ~2–3% across every checkpoint. Only `…-TauSFT-Tau` (and `qwen-base`
run1) still have banking runs from before the drop; **do not add banking
back.**

Each `(model, domain, run)` cell is a `tau2` invocation of `num_trials = 4`
trials over the domain's task set. Task counts and the resulting **expected**
sim count per run:

| Domain | tasks | × trials | = expected sims/run |
|---|---|---|---|
| retail | 114 | 4 | 456 |
| airline | 50 | 4 | 200 |
| telecom | 114 | 4 | 456 |
| mock | 10 | 4 | 40 |
| banking_knowledge | 97 | 4 | 388 |

Runs are repeated seeds: `run1..run3` are the standard 3 seeds (Search `mock`
has extra trials `run4/run5`).

### 2.2 Runner scripts (`eval_scai/tau3/`)

| Script | Purpose |
|---|---|
| `serve_tau3_fleet.sh` | Bring up the 4 agent servers + 2 GLM servers (topology §1.3). |
| `run_tau3.sh` | Main eval driver: exports per-lane judge env, runs `tau2` per `(model, domain, run)` with `--auto-resume`. `mock` is cleared + rerun fresh (unstable task hashes → no `--auto-resume`). |
| `run_wanted.sh` | Finisher for the last outstanding cells (Base-Math telecom run3, Search mock run4/run5) — brings up servers via `bringup_servers_only.sh`, switches GPU2 to Search, runs exactly those 3 in parallel. Do **not** use `run_tau3.sh` for these (it clobbers mock run1-3). |
| `aggregate_tau3.py` | Scored-only aggregator (denominator = *scored* sims). |
| `aggregate_tau3_report.py` | **Report aggregator** (denominator = *expected* sims — the §3 convention). |

Outputs land in `eval_scai/tau3/tau3_rp_results/tau3rp_<model>_<domain>_run<N>/results.json`
(root-owned, mode 0600 — read with `sudo`). Each `results.json` contains full
per-sim trajectories (`simulations[].messages[]`) plus `reward_info`.

### 2.3 Auto-resume & spot recovery (`eval_scai/tau3/aws/`)

Designed to survive AWS spot interruptions:

- `autostart-tau3-wanted.sh` — idempotent full bringup (mount `/mnt/old-data3`
  → set docker data-root → restart docker **after** mount → container → tau2
  install + patches → 6 servers → launch eval). Installed at
  `/usr/local/bin/` and driven by `tau3eval.service`.
- `bringup_servers_only.sh` — steps 0-7 (servers, no eval launch).
- `tau3eval.service` — systemd unit; **`docker` is only in `Wants=`/`After=`,
  not `Requires=`** (a `Requires=docker.service` caused the step-1 docker
  restart to cascade-kill the unit itself).
- `tau2 --auto-resume` skips finished `tau3rp_*` cells and retries
  `infrastructure_error` sims automatically (so judge/GLM-side transient errors
  self-heal on restart; agent-overflow ones re-error and stay unscored).

Two historical bugs fixed in the autostart script: (1) docker-starts-before-mount
race → dockerd bound an empty data-root and the cached image looked missing
(fix: always restart docker after the mount; fail loud if image missing or <6
servers healthy); (2) `banking_knowledge` BM25 cell died on
`ModuleNotFoundError: rank_bm25` (fix: `pip install rank_bm25`).

---

## 3. Report

### 3.1 Metric and denominator convention

**pass^1 = mean binary reward** — the fraction of simulations whose reward is a
pass (`reward ≥ 0.999`).

> **Denominator = full EXPECTED sim count** (`num_tasks × num_trials`; the §2.1
> table). Simulations that never produced a reward (context overflow /
> `infrastructure_error`) are counted as **failures**, not dropped.
>
> ```
> pass^1 = successes / expected        # e.g. 24 / 456 = 5.3%
> ```

This is the conservative, apples-to-apples number. (`aggregate_tau3.py` instead
divides by *scored* sims, which yields an optimistic upper bound — see §3.4.
`aggregate_tau3_report.py` implements the convention documented here.)

Reproduce:

```bash
cd eval_scai/tau3
sudo python3 aggregate_tau3_report.py            # per-run + mean±std, full-expected denom
```

### 3.2 Per-run pass^1 (successes / expected)

**qwen-8b-base**

| Domain | run1 | run2 | run3 |
|---|---|---|---|
| retail | 5.3% (24/456) | 5.3% (24/456) | 4.6% (21/456) |
| airline | 44.0% (88/200) | 38.0% (76/200) | 34.5% (69/200) |
| telecom | 18.6% (85/456) | 16.7% (76/456) | 17.1% (78/456) |
| mock | 5.0% (2/40) | 2.5% (1/40) | 7.5% (3/40) |
| banking_knowledge | 0.8% (3/388) | — | — |

**Qwen3-8B-Base-Math**

| Domain | run1 | run2 | run3 |
|---|---|---|---|
| retail | 2.6% (12/456) | 2.6% (12/456) | 1.1% (5/456) |
| airline | 24.0% (48/200) | 13.5% (27/200) | 7.0% (14/200) |
| telecom | 15.4% (70/456) | 12.7% (58/456) | 11.2% (51/456) |
| mock | 0.0% (0/40) | 2.5% (1/40) | 7.5% (3/40) |

**Qwen3-8B-Base-Math-SeaSFT-Search**

| Domain | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|
| retail | 4.6% (21/456) | 3.3% (15/456) | 2.9% (13/456) | — | — |
| airline | 9.5% (19/200) | 3.5% (7/200) | 4.5% (9/200) | — | — |
| telecom | 13.6% (62/456) | 6.6% (30/456) | 7.2% (33/456) | — | — |
| mock | 42.5% (17/40) | 30.0% (12/40) | 30.0% (12/40) | 22.5% (9/40) | 32.5% (13/40) |

**Qwen3-8B-Base-Math-SeaSFT-Search-TauSFT-Tau** (final)

| Domain | run1 | run2 | run3 |
|---|---|---|---|
| retail | 26.8% (122/456) | 26.1% (119/456) | 27.0% (123/456) |
| airline | 6.5% (13/200) | 7.5% (15/200) | 7.5% (15/200) |
| telecom | 46.3% (211/456) | 48.0% (219/456) | 44.7% (204/456) |
| mock | 75.0% (30/40) | 75.0% (30/40) | 72.5% (29/40) |
| banking_knowledge | 2.1% (8/388) | 2.6% (10/388) | 2.8% (11/388) |

### 3.3 Aggregate pass^1 (mean ± std across runs)

`n` = number of runs averaged.

| Checkpoint | retail | airline | telecom | mock | banking_knowledge |
|---|---|---|---|---|---|
| **qwen-8b-base** | 5.0 ±0.4 (n3) | 38.8 ±4.8 (n3) | 17.5 ±1.0 (n3) | 5.0 ±2.5 (n3) | 0.8 (n1) |
| **Base-Math** | 2.1 ±0.9 (n3) | 14.8 ±8.6 (n3) | 13.1 ±2.1 (n3) | 3.3 ±3.8 (n3) | — |
| **Base-Math-SeaSFT-Search** | 3.6 ±0.9 (n3) | 5.8 ±3.2 (n3) | 9.1 ±3.9 (n3) | 30.8 ±1.4 (n3)¹ | — |
| **…-Search-TauSFT-Tau** (final) | 26.6 ±0.5 (n3) | 7.2 ±0.6 (n3) | 46.3 ±1.6 (n3) | 74.2 ±1.4 (n3) | 2.5 ±0.4 (n3) |

*(all values %; user sim = GLM-4.7-Flash; denominator = full expected count.)*

¹ **Search `mock`** is averaged over the three consistent trials
**run2/run3/run5 (30.0, 30.0, 32.5) → 30.8 ±1.4**; the outlier trials run1
(42.5) and run4 (22.5) are excluded. Averaging all five runs instead gives
31.5 ±7.2 (n5), as `aggregate_tau3_report.py` prints by default.

### 3.4 Interpretation & caveats

- **Coverage bias.** The denominator convention matters because coverage is
  uneven. For the first three checkpoints many of the longest retail/telecom
  sims overflow the agent's 32K context and never score (§1.5). The
  full-expected denominator here treats those as failures — the **conservative
  lower bound**. The scored-only `aggregate_tau3.py` divides by completed sims
  and gives an **optimistic upper bound**. Truth is between the two for
  low-coverage cells; e.g. Base-Math airline run3 is 7.0% (full) vs 31.1%
  (scored).
- **The final checkpoint is trustworthy.** `…-TauSFT-Tau` runs at ~100%
  coverage, so its full-denominator numbers ≈ its scored numbers — retail
  ~26.6, telecom ~46.3, mock ~74.2 are real. It is the strongest checkpoint on
  the target agentic domains.
- **Stage progression.** Search unlocks `mock` (base ~5 → Search ~31 → TauSFT
  ~74); TauSFT-Tau drives large gains on retail (→26.6) and telecom (→46.3).
- **airline regresses** on the final checkpoint (base 38.8 → final 7.2). This is
  real in both denominator conventions (airline runs at ~100% coverage), and is
  worth investigating — likely over-specialization to the retail/telecom/mock
  tool-use style at airline's expense.
- **banking is at the RAG floor** (~1–3%) for every checkpoint — the reason it
  was dropped from the default domain set.

### 3.5 Trajectories

Every scored simulation's full trajectory is stored inline in each
`results.json` under `simulations[].messages[]` (role / content / tool_calls /
usage / cost / timestamps, plus τ3 voice fields), alongside `reward_info`,
`termination_reason`, `trial`, `seed`, etc. Across the 54 result cells there are
**15,456 total trajectories**. Files are root-owned `0600` — read with `sudo`.
