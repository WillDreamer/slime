#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# tau-bench (retail) GRPO RL for Qwen3.5-4B-Base on Greenland — multi-node (_mns)
#
# Derived from examples/math_reasoning/run_qwen35_4b_base_mns.sh (the VALIDATED
# Greenland multinode launcher). The Ray topology, colocate/disaggregated
# env-gating, the GPU-registration wait loop, and the EFA/NCCL runtime-env are
# kept byte-for-byte from that script. What changed for tau-bench:
#   * model checkpoints -> Qwen3.5-4B-Base
#   * ROLLOUT/EVAL args -> tau-bench retail task indices (not math prompts)
#   * CUSTOM_ARGS -> --custom-generate-function-path generate_with_tau.generate
#     (multi-turn tool-use rollout against the in-cluster SGLang server)
#   * user simulator -> selected by TAU_USER_STRATEGY (default "local"):
#       local  — a frozen model (GLM-4.7-Flash) served IN-CLUSTER by slime as the
#                "user_sim" entry of a GENERATED multi-model --sglang-config. The
#                rollout resolves its router URL at run time. NO external egress.
#                GPU split is driven by USER_SIM_NUM_GPUS (= --user-sim-nodes*8).
#       claude — Bedrock Claude (boto3, cross-region). TAU_*/AWS_* env injected.
#
# Submit — LOCAL user-sim (6 nodes = 1 train + 5 rollout; --user-sim-nodes 1 =>
# rollout splits into 32 GPU actor + 8 GPU GLM user_sim = 40 = --rollout-nodes*8):
#   python3 greenland_cli_mns.py obx \
#     --script examples/tau-bench/run_qwen35_4b_tau_mns.sh \
#     --num-nodes 6 --rollout-nodes 5 --user-sim-nodes 1 \
#     --stage-model Qwen3.5/Qwen3.5-4B-Base/ \
#     --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ \
#     --stage-model GLM/GLM-4.7-Flash/ \
#     --stage-data tau-bench/
#
# Submit — Bedrock Claude user-sim (colocate, 1 node = 8 GPU; no user-sim node):
#   python3 greenland_cli_mns.py obx --env TAU_USER_STRATEGY=claude --user-sim-nodes 0 \
#     --script examples/tau-bench/run_qwen35_4b_tau_mns.sh --num-nodes 1 \
#     --stage-model Qwen3.5/Qwen3.5-4B-Base/ \
#     --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ \
#     --stage-data tau-bench/
# ─────────────────────────────────────────────────────────────────────────────

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Overridable via env (Greenland OBX bootstrap sets these to the container's
# local NVMe mirror; defaults keep dev-box behaviour unchanged).
ROOT_DIR=${ROOT_DIR:-/data2/whx}
MODEL_ROOT=${MODEL_ROOT:-/data2/whx/models}
DATA_ROOT=${DATA_ROOT:-/data2/whx/data}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-4B.sh"
WANDB_API_KEY="${WANDB_API_KEY}"

GPU_LIST=(0 1 2 3 4 5 6 7)
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_LIST[*]}")
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}
echo "Detected ${NUM_GPUS} GPUs for this run"

ROLLOUT_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256
# _nothink: enable_thinking=False fix (Qwen3.5 thinking template was breaking the
# loss-mask alignment -> raw_reward declined). Fresh group so the new behavior's
# metrics + rollout-debug dumps don't mix with the old thinking-on run.
WANDB_GROUP="tau_rl_Qwen35-4B_nothink_bs_${ROLLOUT_BATCH_SIZE}"

# ── tau-bench user simulator ──
# Selected by TAU_USER_STRATEGY. Two paths are supported:
#   "local"  (default here) — a model served IN-CLUSTER over an OpenAI-compatible
#             HTTP endpoint (GLM-flash on a dedicated node, launched by slime as
#             the "user_sim" model in user_sim_sglang.yaml). No external egress;
#             the endpoint is resolved at run time from the live SGLang router by
#             generate_with_tau.py and published via TAU_USER_SIM_URL. Read in
#             tau_bench/envs/user.py::LocalUserSimulationEnv.
#   "claude" — Bedrock Claude (boto3, cross-region). Read in
#             tau_bench/envs/user.py::BedrockClaudeUserSimulationEnv.
# Exported here AND mirrored into RUNTIME_ENV_JSON so rollout actors on worker
# nodes inherit them (the Ray worker shell env does NOT propagate to job tasks).
#   TAU_USER_MODEL_ID  served model name (local; default "user_sim", MUST match
#                      served_model_name in user_sim_sglang.yaml) OR Bedrock id (claude)
#   TAU_USER_SIM_MODEL named model in --sglang-config to route to (local; default "user_sim")
#   TAU_TOOL_PARSER    qwen3_coder — MUST match Qwen3.5's chat-template tool
#                      format (<function=..><parameter=..>). Mismatch => every
#                      turn parses as RESPOND and training collapses.
export TAU_USER_STRATEGY="${TAU_USER_STRATEGY:-local}"
export TAU_ENV="${TAU_ENV:-retail}"
export TAU_TASK_SPLIT="${TAU_TASK_SPLIT:-train}"
export TAU_TOOL_PARSER="${TAU_TOOL_PARSER:-qwen3_coder}"

if [ "${TAU_USER_STRATEGY}" = "claude" ]; then
    # Bedrock Claude user-sim (cross-region; needs the boto3 credential chain).
    export TAU_USER_MODEL_ID="${TAU_USER_MODEL_ID:-us.anthropic.claude-opus-4-7}"
    export TAU_BEDROCK_REGION="${TAU_BEDROCK_REGION:-us-east-1}"
    # AWS config for the in-container boto3 default credential chain. The bootstrap
    # writes /root/.aws/config (EcsContainer -> assume greenland-dev-role) and sets
    # AWS_CONFIG_FILE; default both here so a dev-box run still works with a profile.
    export AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-/root/.aws/config}"
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${TAU_BEDROCK_REGION}}"

    # ── Bedrock egress self-test (LOUD) ──
    # The job runs in ap-south-1 and calls Bedrock in us-east-1; probe it once up
    # front so a network block shows up here, not as silent rollout aborts later.
    python3 - <<'PYEOF' || echo "WARN: Bedrock self-test failed — user-sim calls will likely abort; check egress to bedrock-runtime.${TAU_BEDROCK_REGION}.amazonaws.com"
import json, os, sys
try:
    import boto3
    from botocore.config import Config
    region = os.environ.get("TAU_BEDROCK_REGION", "us-east-1")
    model = os.environ.get("TAU_USER_MODEL_ID", "us.anthropic.claude-opus-4-7")
    cli = boto3.client("bedrock-runtime", region_name=region,
                       config=Config(retries={"total_max_attempts": 2, "mode": "standard"},
                                     connect_timeout=10, read_timeout=30))
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say OK."}]}
    try:
        r = cli.invoke_model(modelId=model, body=json.dumps(body),
                             contentType="application/json", accept="application/json")
    except Exception as e:
        if "temperature" in str(e).lower():
            r = cli.invoke_model(modelId=model, body=json.dumps(body),
                                 contentType="application/json", accept="application/json")
        else:
            raise
    out = json.loads(r["body"].read())
    txt = "".join(b.get("text", "") for b in out.get("content", []))
    print(f"[tau egress self-test] Bedrock {model} @ {region} OK -> {txt!r}")
except Exception as e:
    print(f"[tau egress self-test] FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
else
    # LOCAL user-sim (e.g. GLM-flash). No external egress, no AWS creds needed.
    # TAU_USER_SIM_URL is filled at run time from the live router; leave it unset
    # here unless you are pointing at a self-launched / fixed endpoint.
    export TAU_USER_MODEL_ID="${TAU_USER_MODEL_ID:-user_sim}"
    export TAU_USER_SIM_MODEL="${TAU_USER_SIM_MODEL:-user_sim}"
    echo "[tau] LOCAL user-sim: model '${TAU_USER_MODEL_ID}', router model '${TAU_USER_SIM_MODEL}' (endpoint resolved from SGLang at run time)"
fi

# Disaggregated vs colocate memory budgets (same logic as the math _mns script).
# ROLLOUT_NUM_GPUS is exported by the Greenland bootstrap (0 / unset = colocate).
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    # 15360 OOM'd在训练侧 logits 步(base job 962645): empty_strided_cuda((s10,1,248320),
    # fp32) 要 15.28GiB 分不出(vocab=248320 巨大,fp32 logits 峰值 ∝ max-tokens)。
    # 降回 9216(同模型同 vocab,OOM 同理),对齐 base _mns 脚本。
    MAX_TOKENS_PER_GPU=9216
    SGLANG_MEM_FRACTION=0.85
else
    MAX_TOKENS_PER_GPU=9216
    SGLANG_MEM_FRACTION=0.7
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base/
   --ref-load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base_torch_dist/
   # FROM SCRATCH: --load points at the BASE torch_dist (the SFT/base init), and
   # --save is a FRESH dir (-nothink). slime resumes from --save only if it holds
   # a newer checkpoint; a fresh --save dir => clean start from --load (the base),
   # so the no-think behavior is trained from the base, not resumed off the old
   # thinking-on run. (Matches the math _mns from-scratch pattern.)
   --load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base_torch_dist/
   --save ${MODEL_ROOT}/AECE/Qwen3.5-4B-Base-Tau-nothink/
   # Save every 5 steps; permanently RETAIN every 20th (--save-retain-interval
   # must be a multiple of --save-interval, validated by Megatron). The
   # non-retained saves roll with a window of 1, so disk holds the latest frequent
   # save + the retained milestones (steps 20/40/60...). --save-retain-interval is
   # Megatron's native arg (do NOT re-add it in slime/utils/arguments.py — that
   # dup crashed job c9b3d8dc); train.py reads it via getattr to prune + mirror to S3.
   --save-interval 5
   --save-retain-interval 20
)

# Save every rollout's samples (.pt) for offline inspection / SFT distillation.
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/AECE/tau_rl/${WANDB_GROUP}"

ROLLOUT_ARGS=(
   --prompt-data "${DATA_ROOT}/tau-bench/retail_train_tasks.jsonl"
   --input-key index
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
   --log-passrate
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data retail-dev "${DATA_ROOT}/tau-bench/retail_dev_tasks.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 2048
   --eval-top-k 1
)

PERF_ARGS=(
   # TP=2 (was 1): job 812f6961 OOM'd on train step 4 at the entropy clone with
   # only ~940MiB short (GPU 0 at 126.85/139.72 GiB). The [T, V=248320] fp32 logits
   # head is the OOM main term; TP=2 shards vocab -> halves that peak. TP ceiling is
   # 4 here (--num-query-groups 4 must be divisible by TP). The 24 GatedDeltaNet
   # linear-attn layers are duplicated (not sharded) under TP — correct but no mem
   # saving there; the logits head IS sharded, which is what matters for this OOM.
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}

   # CRITICAL for tau (job 662143 OOM'd here, GPU3 tried to alloc 21.08GiB at the
   # entropy step). tau is MULTI-TURN so a trajectory accumulates to ~11.5k tokens
   # (rollout/total_lengths=11557); the training forward then holds a full
   # [T, V=248320] fp32 logits tensor, and slime's loss computes log-probs+entropy
   # on the WHOLE T at once when --log-probs-chunk-size is unset (default -1).
   # compute_entropy_from_logits does logits.clone() PLUS _VocabParallelEntropy
   # allocates another full [T,V] (`vocab_parallel_logits - logits_max`) -> the
   # ~21GiB doubling that OOM'd (and 50GiB was reserved-but-fragmented). TP=1 means
   # the giant vocab isn't sharded, so there's nothing to split it across.
   # Chunking the log-prob/entropy compute along the TOKEN dim bounds the peak
   # buffer to [chunk, V] (~1GiB at 1024) — numerically identical (per-token values
   # are independent), and does NOT touch rollout-batch-size / seq-len /
   # max-tokens-per-gpu. The math run never hit this (single-turn -> shorter T).
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   # TIS: recommended for the multi-turn tool-use rollout (train/infer mismatch)
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project AECE
   --wandb-group ${WANDB_GROUP}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

# ── Router policy / session affinity (the new knob) ──
# TAU_ROUTER_POLICY selects the SGLang router's load-balancing policy:
#   cache_aware        (default) — route by live prefix-cache state. Unchanged
#                       behaviour; this is the router's own implicit default.
#   consistent_hashing            — SESSION AFFINITY for the multi-turn rollout:
#                       all turns of one trajectory share an X-SMG-Routing-Key
#                       (set in trainable_agents.asolve) and hash to the SAME
#                       worker, so each turn reuses that worker's prefix cache
#                       instead of re-prefilling the whole growing history. This
#                       targets the prefill cost that EAGLE/MTP does NOT speed up.
#   round_robin                   — ignore cache (diagnostic).
# Carried into the actor request path via --router-policy (a real slime arg now;
# it drives both the router launch AND the per-request routing-key header).
export TAU_ROUTER_POLICY="${TAU_ROUTER_POLICY:-cache_aware}"

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
   --sglang-server-concurrency 1024
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # custom all-reduce CUDA-graph capture fails on this H200/driver -> disable
   --sglang-disable-custom-all-reduce
   --router-policy ${TAU_ROUTER_POLICY}
)
# LOCAL user-sim: deploy a 2nd (frozen) user-simulation model (GLM-4.7-Flash)
# alongside the actor via a multi-model --sglang-config. The YAML is GENERATED
# here because the GPU split depends on USER_SIM_NUM_GPUS (= --user-sim-nodes*8,
# exported by the Greenland bootstrap; default 8 = one node for a dev-box run).
#   actor model    = ROLLOUT_NUM_GPUS - USER_SIM_NUM_GPUS  (TP=4, inherits SGLANG_ARGS)
#   user_sim model = USER_SIM_NUM_GPUS                      (GLM-4.7-Flash, TP=4, frozen)
# The two MUST sum to ROLLOUT_NUM_GPUS (slime validates this at rollout.py:1216).
# GLM-4.7-Flash SGLang config follows the model's HF page (zai-org/GLM-4.7-Flash):
# tp-size 4, tool-call-parser glm47, reasoning-parser glm45, served-model-name,
# mem-fraction-static 0.8. Speculative (EAGLE/MTP) is intentionally OMITTED for
# the user-sim (simpler/robust; can be added later via overrides).
USER_SIM_NUM_GPUS=${USER_SIM_NUM_GPUS:-0}
if [ "${TAU_USER_STRATEGY}" = "local" ]; then
   if [ "${USER_SIM_NUM_GPUS:-0}" -le 0 ] || [ "${ROLLOUT_NUM_GPUS:-0}" -le 0 ]; then
      echo "FATAL: local user-sim needs disaggregated rollout with reserved GPUs " \
           "(ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}, USER_SIM_NUM_GPUS=${USER_SIM_NUM_GPUS:-0}). " \
           "Submit with --rollout-nodes > --user-sim-nodes >= 1." >&2
      exit 1
   fi
   ACTOR_ROLLOUT_GPUS=$(( ROLLOUT_NUM_GPUS - USER_SIM_NUM_GPUS ))
   if [ "${ACTOR_ROLLOUT_GPUS}" -le 0 ]; then
      echo "FATAL: no actor rollout GPUs left (ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS}, " \
           "USER_SIM_NUM_GPUS=${USER_SIM_NUM_GPUS}). Need --rollout-nodes > --user-sim-nodes." >&2
      exit 1
   fi
   # GLM-4.7-Flash TP. The HF page recommends tp-size 4; clamp to the user-sim
   # pool so a single reserved node (8 GPU) yields 2 engines @ TP=4, and a tiny
   # pool still launches at least one engine.
   USER_SIM_TP=${USER_SIM_TP:-4}
   if [ "${USER_SIM_NUM_GPUS}" -lt "${USER_SIM_TP}" ]; then USER_SIM_TP=${USER_SIM_NUM_GPUS}; fi
   USER_SIM_MODEL_PATH="${USER_SIM_MODEL_PATH:-${MODEL_ROOT}/GLM/GLM-4.7-Flash}"
   USER_SIM_SERVED_NAME="${TAU_USER_MODEL_ID:-user_sim}"
   USER_SIM_YAML="${SCRIPT_DIR}/user_sim_sglang.generated.yaml"
   cat > "${USER_SIM_YAML}" <<YAMLEOF
# AUTO-GENERATED by run_qwen35_4b_tau_mns.sh — do not edit by hand.
# rollout split: actor=${ACTOR_ROLLOUT_GPUS} GPU + user_sim=${USER_SIM_NUM_GPUS} GPU = ${ROLLOUT_NUM_GPUS} (= --rollout-num-gpus)
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: ${ACTOR_ROLLOUT_GPUS}
  - name: user_sim
    model_path: ${USER_SIM_MODEL_PATH}
    update_weights: false
    num_gpus_per_engine: ${USER_SIM_TP}
    server_groups:
      - worker_type: regular
        num_gpus: ${USER_SIM_NUM_GPUS}
        overrides:
          served_model_name: ${USER_SIM_SERVED_NAME}
          tool_call_parser: glm47
          reasoning_parser: glm45
          mem_fraction_static: 0.8
          trust_remote_code: true
YAMLEOF
   echo "[tau] generated multi-model sglang config -> ${USER_SIM_YAML}:"
   cat "${USER_SIM_YAML}"
   SGLANG_ARGS+=( --sglang-config "${USER_SIM_YAML}" )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # 关闭 bias-dropout fusion:Megatron 的 bias_dropout_add_fused_train 带 @jit_fuser
   # (torch>=2.2 即 torch.compile)。dropout=0 时融的是 no-op、零收益;而
   # --use-dynamic-batch-size 下每 microbatch 形状不同 → torch.compile 每步重编译,
   # 撞 recompile_limit(8) 后训练退化到 ~42s/microbatch 并卡死(base job 605065 即此)。
   # 关掉走纯 Python 路径,根除 recompile。对齐 base _mns 脚本。
   --no-bias-dropout-fusion
)

CUSTOM_ARGS=(
   # multi-turn tool-use rollout that drives the tau-bench env + Bedrock user sim
   --custom-generate-function-path generate_with_tau.generate
   # TIS-related args (recommended to enable when using --use-tis)
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Multi-node: the Greenland bootstrap exports MASTER_ADDR (= head IP) and
# ACTOR_NUM_NODES / ROLLOUT_NUM_GPUS. Single-node/dev-box defaults preserved.
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}
# tau-bench example modules live in SCRIPT_DIR; they import the vendored
# tau_bench package (also under SCRIPT_DIR) and slime. Put SCRIPT_DIR on the path.
export PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"

# ── Generate tau-bench task data (per original example README) ──
# The original example seeds training data with:
#     cd examples/tau-bench && python tau1_mock.py --local_dir <dir>
# which writes retail_{train,test,dev}_tasks.jsonl (full task metadata, one JSON
# per line: {"index": i, "metadata": {...}}). tau1_mock uses user_strategy=human
# so it needs NO LLM / no boto3 — just the vendored tau_bench package on
# PYTHONPATH (set above). We run it HERE so the data is always generated by the
# same vendored tau_bench version the rollout uses, into the path --prompt-data
# expects, with no dependence on what happens to be pre-staged in S3.
#
# Only the MAIN node reaches this point (the Greenland bootstrap blocks worker
# nodes in `ray start --block` and exits them before the run script), and
# train.py loads --prompt-data on the head, so this runs exactly once.
# Idempotent: skip if the train split already exists (e.g. staged via
# `--stage-data tau-bench/`), so a re-run doesn't waste time.
TAU_DATA_DIR="${DATA_ROOT}/tau-bench"
mkdir -p "${TAU_DATA_DIR}"
if [ -s "${TAU_DATA_DIR}/retail_${TAU_TASK_SPLIT}_tasks.jsonl" ]; then
    echo "[tau data] ${TAU_DATA_DIR}/retail_${TAU_TASK_SPLIT}_tasks.jsonl already present; skipping generation"
else
    echo "[tau data] generating retail_{train,test,dev}_tasks.jsonl via tau1_mock.py -> ${TAU_DATA_DIR}"
    ( cd "${SCRIPT_DIR}" && python3 tau1_mock.py --local_dir "${TAU_DATA_DIR}" )
fi
ls -l "${TAU_DATA_DIR}"/retail_*_tasks.jsonl

# Dev-box needs the system lib dir prepended; the Greenland image sets
# SKIP_SYS_LDPATH=1 (prepending breaks libcudnn there).
if [ -z "${SKIP_SYS_LDPATH:-}" ]; then
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
fi
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir ${RAY_TEMP_DIR:-${ROOT_DIR}/ray_temp}

# Multi-node: wait until ALL GPUs (training + rollout) register with Ray before
# submitting (query GCS via ray.cluster_resources(), not `ray status` text).
#
# HARDENING (learned from a live 2-node failure):
#  * AWS Batch brings the CHILD node up MINUTES after the main node (observed:
#    head ready at T+0, worker didn't start booting until ~T+5min). So the wait
#    must tolerate a long boot-skew — bumped to 180 iters x 10s = 30 min.
#  * The Greenland stuck-job detector killed the head at ~100-150s because every
#    wait line was the IDENTICAL string "8/16 GPUs registered" (looks like a hung
#    app). We now print VARYING content every iteration — wall-clock, elapsed
#    seconds, GPU count, AND the list of alive node IPs (which changes as nodes
#    join) — so the detector sees forward progress, and the log is diagnosable.
#  * On timeout we FAIL LOUDLY with a cross-subnet diagnostic instead of falling
#    through to `ray job submit` (which would then hang/err deep in train.py's
#    placement-group with a far more confusing message).
EXPECTED_GPUS=$(( ACTOR_NUM_NODES * NUM_GPUS + ROLLOUT_NUM_GPUS ))
if [ "$EXPECTED_GPUS" -gt "$NUM_GPUS" ]; then
    echo "Waiting for ${EXPECTED_GPUS} GPUs (train $(( ACTOR_NUM_NODES * NUM_GPUS )) + rollout ${ROLLOUT_NUM_GPUS}) to register with Ray (head=${MASTER_ADDR}, up to 30 min for child-node boot-skew)..."
    WAIT_OK=0
    WAIT_START=$(date +%s)
    for i in $(seq 1 180); do
        # Query GPU count AND the alive node IPs in one ray.init (alive list
        # changes as workers join -> varying output that reads as progress).
        READ=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
r = ray.cluster_resources()
alive = sorted(n['NodeManagerAddress'] for n in ray.nodes() if n.get('Alive'))
print(int(r.get('GPU', 0)))
print(len(alive))
print(','.join(alive))
ray.shutdown()
" 2>/dev/null)
        GOT=$(echo "$READ" | sed -n '1p'); GOT=${GOT:-0}
        NNODES=$(echo "$READ" | sed -n '2p'); NNODES=${NNODES:-0}
        ALIVE_IPS=$(echo "$READ" | sed -n '3p')
        ELAPSED=$(( $(date +%s) - WAIT_START ))
        echo "  [$(date -u +%H:%M:%S) | +${ELAPSED}s | iter $i] ${GOT}/${EXPECTED_GPUS} GPUs, ${NNODES} node(s) alive: [${ALIVE_IPS}]"
        [ "$GOT" -ge "$EXPECTED_GPUS" ] && { WAIT_OK=1; break; }
        sleep 10
    done
    if [ "$WAIT_OK" != "1" ]; then
        HEAD_SUBNET="${MASTER_ADDR%.*}"
        echo "FATAL: only ${GOT}/${EXPECTED_GPUS} GPUs registered after $(( $(date +%s) - WAIT_START ))s."
        echo "       head=${MASTER_ADDR} (/.24=${HEAD_SUBNET}); alive nodes: [${ALIVE_IPS}]"
        echo "       If only the head's ${NUM_GPUS} GPUs ever showed, the worker node never joined the"
        echo "       Ray cluster — most likely the control-plane port 6379 is blocked across subnets"
        echo "       (worker on a different /24 than ${HEAD_SUBNET}.x), or the child node never booted."
        echo "       Check the WORKER node's CloudWatch stream for '[bootstrap] worker N: ...' lines"
        echo "       (it now logs GCS-reachability probes + join verification)."
        ray stop --force 2>/dev/null || true
        exit 1
    fi
    echo "All ${EXPECTED_GPUS} GPUs registered in $(( $(date +%s) - WAIT_START ))s; proceeding to submit."
fi

# Multi-node EFA/socket env (same as the math _mns script). Actors on worker
# nodes inherit the JOB runtime-env, so socket-iface pins + EFA must live here.
MULTINODE_ENV=""
if [ "${ACTOR_NUM_NODES:-1}" -gt 1 ]; then
    MULTINODE_ENV=",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-eth0}\",
    \"TP_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"FI_PROVIDER\": \"${FI_PROVIDER:-efa}\",
    \"FI_EFA_USE_DEVICE_RDMA\": \"1\",
    \"NCCL_DEBUG\": \"${NCCL_DEBUG:-INFO}\",
    \"NCCL_DEBUG_SUBSYS\": \"INIT,NET\",
    \"TORCH_NCCL_BLOCKING_WAIT\": \"1\",
    \"TORCH_NCCL_TIMEOUT_MS\": \"${TORCH_NCCL_TIMEOUT_MS:-600000}\",
    \"NCCL_ASYNC_ERROR_HANDLING\": \"1\",
    \"NCCL_NET_PLUGIN\": \"${NCCL_NET_PLUGIN:-/opt/amazon/ofi-nccl/lib/libnccl-net.so}\",
    \"LD_LIBRARY_PATH\": \"/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64}\""
fi

# tau-bench rollout actors run the user simulator, so EVERY actor (on every node)
# needs the tau env. Inject it into the Ray job runtime-env so worker-node actors
# inherit it too (the Ray worker shell env does NOT propagate to job tasks).
# Common tau vars (both strategies):
TAU_ENV_JSON="
    \"TAU_USER_MODEL_ID\": \"${TAU_USER_MODEL_ID}\",
    \"TAU_USER_STRATEGY\": \"${TAU_USER_STRATEGY}\",
    \"TAU_ENV\": \"${TAU_ENV}\",
    \"TAU_TASK_SPLIT\": \"${TAU_TASK_SPLIT}\",
    \"TAU_TOOL_PARSER\": \"${TAU_TOOL_PARSER}\""
if [ "${TAU_USER_STRATEGY}" = "claude" ]; then
    # Bedrock path: forward AWS creds config + region so worker-node boto3 can
    # resolve the same EcsContainer -> greenland-dev-role chain.
    TAU_ENV_JSON="${TAU_ENV_JSON},
    \"TAU_BEDROCK_REGION\": \"${TAU_BEDROCK_REGION}\",
    \"AWS_CONFIG_FILE\": \"${AWS_CONFIG_FILE}\",
    \"AWS_DEFAULT_REGION\": \"${AWS_DEFAULT_REGION}\",
    \"AWS_REGION\": \"${AWS_DEFAULT_REGION}\""
    if [ -n "${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-}" ]; then
        TAU_ENV_JSON="${TAU_ENV_JSON},
    \"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI\": \"${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI}\""
    fi
else
    # Local path: the named-model router to resolve at run time. (TAU_USER_SIM_URL
    # is filled per-actor by generate_with_tau.py from args, not injected here.)
    TAU_ENV_JSON="${TAU_ENV_JSON},
    \"TAU_USER_SIM_MODEL\": \"${TAU_USER_SIM_MODEL}\""
    # Forward an explicit endpoint only if the operator pinned one.
    if [ -n "${TAU_USER_SIM_URL:-}" ]; then
        TAU_ENV_JSON="${TAU_ENV_JSON},
    \"TAU_USER_SIM_URL\": \"${TAU_USER_SIM_URL}\""
    fi
fi

# FI_EFA_FORK_SAFE=1 is UNCONDITIONAL (not gated on ACTOR_NUM_NODES>1 like the
# other FI_* knobs in MULTINODE_ENV): the EFA image bakes FI_PROVIDER=efa into its
# global ENV, so EVERY process initializes the EFA/libfabric provider regardless
# of node count. Megatron's async checkpoint save fork()s worker procs, and EFA
# SIGABRTs the whole job on the FIRST save unless fork-safe is set (math job
# 910787d4 died this way at ~11.8h). Critically, tau's user-sim topology
# (--num-nodes 6 --rollout-nodes 5) has ACTOR_NUM_NODES=1, so MULTINODE_ENV is
# EMPTY and would NOT carry this — it must live here, in the always-on env_vars.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    ${TAU_ENV_JSON}${MULTINODE_ENV}
  }
}"

# Colocate vs disaggregated resource args (same as the math _mns script).
RESOURCE_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS}
   --num-gpus-per-node ${NUM_GPUS}
)
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    RESOURCE_ARGS+=( --rollout-num-gpus ${ROLLOUT_NUM_GPUS} )
    echo "Disaggregated: actor=$(( ACTOR_NUM_NODES * NUM_GPUS )) GPU, rollout=${ROLLOUT_NUM_GPUS} GPU (no colocate)"
else
    RESOURCE_ARGS+=( --colocate )
    echo "Colocate: actor and rollout share $(( ACTOR_NUM_NODES * NUM_GPUS )) GPU"
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${RESOURCE_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
