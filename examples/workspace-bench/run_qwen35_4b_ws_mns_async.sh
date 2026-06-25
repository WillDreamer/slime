#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Workspace-Bench GRPO RL for Qwen3.5-4B-Base on Greenland — ASYNC multi-node
#
# Migrated from examples/tau-bench/run_qwen35_4b_tau_mns_async.sh. Same Greenland
# scaffold (Ray job submit, disaggregated async overlap, EFA/NCCL runtime-env,
# hardened GPU-registration wait, FI_EFA_FORK_SAFE, TP=2, --log-probs-chunk-size,
# EAGLE/MTP, TIS). What changed for Workspace-Bench:
#   * SINGLE MODEL: there is NO user simulator (Workspace-Bench is single-shot
#     agentic), so the entire multi-model --sglang-config / USER_SIM_* machinery
#     is gone. The actor is the only rollout model -> EAGLE goes into the global
#     SGLANG_ARGS (the simple branch), which also re-enables spec_accept_rate.
#   * REWARD = a Bedrock Claude RUBRIC JUDGE (judge.py), called once at the end of
#     each trajectory inside the env's terminal `finish` step. We keep tau's AWS
#     credential forwarding (EcsContainer -> greenland-dev-role) — now for the
#     judge instead of a user-sim — and repurpose the loud Bedrock egress
#     self-test to probe WS_JUDGE_MODEL_ID before any rollout runs.
#   * FILE-EDITING agent: --custom-generate-function-path generate_with_ws.generate;
#     the rollout drives a workspace env with file tools (read/write/edit/grep/finish).
#   * DATA: prepared + staged by prepare_ws_data.py to s3://whx-agent/data/workspace-bench/,
#     pulled via --stage-data workspace-bench/. The env locates each task's
#     metadata.json + persona workspace at rollout time.
#
# Async REQUIRES disaggregated (train_async.py asserts not colocate), so submit
# with --rollout-nodes > 0 (and --user-sim-nodes 0 — there is no user sim).
#
# Submit (e.g. 3 nodes = 1 train + 2 rollout):
#   python3 greenland_cli_mns_async.py obx \
#     --script examples/workspace-bench/run_qwen35_4b_ws_mns_async.sh \
#     --num-nodes 3 --rollout-nodes 2 --user-sim-nodes 0 \
#     --stage-model Qwen3.5/Qwen3.5-4B-Base/ \
#     --stage-model Qwen3.5/Qwen3.5-4B-Base_torch_dist/ \
#     --stage-data workspace-bench/
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

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
WANDB_GROUP="ws_Qwen35-4B_async_bs_${ROLLOUT_BATCH_SIZE}"

# train_async.py REQUIRES disaggregated (it asserts not args.colocate). The
# Greenland bootstrap exports ROLLOUT_NUM_GPUS>0 only when submitted with
# --rollout-nodes>0. Fail fast with a clear message if someone runs this async
# script in a colocate (single-pool) topology.
if [ "${ROLLOUT_NUM_GPUS:-0}" -le 0 ]; then
    echo "FATAL: the ASYNC script requires DISAGGREGATED mode (train_async.py asserts"
    echo "       not colocate). Submit with greenland_cli_mns_async.py and --rollout-nodes>0,"
    echo "       e.g. --num-nodes 3 --rollout-nodes 2. (ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-unset})"
    exit 1
fi

# ── Workspace-Bench rollout knobs ──
# Exported here AND mirrored into RUNTIME_ENV_JSON so rollout actors on worker
# nodes inherit them (the Ray worker shell env does NOT propagate to job tasks).
#   WS_SPLIT        lite | full — chooses the staged data subset + jsonl name.
#   WS_TASK_SPLIT   train | dev — which index-map slice the env iterates.
#   WS_TOOL_PARSER  qwen3_coder — MUST match Qwen3.5's chat-template tool format
#                   (<function=..><parameter=..>). Mismatch => every turn parses
#                   as RESPOND and training collapses.
#   WS_ENABLE_THINKING / WS_STRIP_HISTORICAL_THINK — same think machinery as tau
#                   (the format regex + historical-think strip were tuned there;
#                   watch rollout/frac_trained early — if it collapses, set
#                   WS_ENABLE_THINKING=0; see memory slime-think-token-loss-mask).
export WS_ENV="${WS_ENV:-workspace}"
export WS_SPLIT="${WS_SPLIT:-full}"
export WS_TASK_SPLIT="${WS_TASK_SPLIT:-train}"
export WS_TOOL_PARSER="${WS_TOOL_PARSER:-qwen3_coder}"
export WS_ENABLE_THINKING="${WS_ENABLE_THINKING:-1}"
export WS_STRIP_HISTORICAL_THINK="${WS_STRIP_HISTORICAL_THINK:-1}"
export WS_ENV_THREAD_WORKERS="${WS_ENV_THREAD_WORKERS:-256}"
export WS_MAX_STEPS="${WS_MAX_STEPS:-30}"
export WS_READ_CHAR_CAP="${WS_READ_CHAR_CAP:-8192}"
export WS_DATA_DIR="${WS_DATA_DIR:-${DATA_ROOT}/workspace-bench}"
export WS_ROLLOUT_ROOT="${WS_ROLLOUT_ROOT:-${ROOT_DIR}/ws_rollouts}"

# ── Rubric judge (the reward) ──
# Bedrock Claude, default Sonnet 4.6 (strong grader, accepts temperature=0, far
# cheaper/faster than opus at rollout-batch scale). The boto3 default credential
# chain resolves the same EcsContainer -> greenland-dev-role the tau Bedrock path
# used; on a dev box set WS_JUDGE_PROFILE=greenland-dev.
export WS_JUDGE_MODEL_ID="${WS_JUDGE_MODEL_ID:-us.anthropic.claude-sonnet-4-6}"
export WS_BEDROCK_REGION="${WS_BEDROCK_REGION:-us-east-1}"
export WS_JUDGE_MAX_TOKENS="${WS_JUDGE_MAX_TOKENS:-4096}"
export WS_JUDGE_FILE_CHARS="${WS_JUDGE_FILE_CHARS:-12000}"
# AWS config for the in-container boto3 default credential chain. The bootstrap
# writes /root/.aws/config (EcsContainer -> assume greenland-dev-role) and sets
# AWS_CONFIG_FILE; default both here so a dev-box run still works with a profile.
export AWS_CONFIG_FILE="${AWS_CONFIG_FILE:-/root/.aws/config}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${WS_BEDROCK_REGION}}"

# ── Bedrock judge egress self-test (LOUD) ──
# The job runs in ap-south-1 and calls Bedrock in us-east-1; probe the JUDGE model
# once up front so a network/permission block shows up here, not as every rollout
# silently scoring reward=0 later.
python3 - <<'PYEOF' || echo "WARN: Bedrock judge self-test failed — ALL rewards will be 0; check egress to bedrock-runtime.${WS_BEDROCK_REGION}.amazonaws.com and greenland-dev-role Bedrock access"
import json, os, sys
try:
    import boto3
    from botocore.config import Config
    region = os.environ.get("WS_BEDROCK_REGION", "us-east-1")
    model = os.environ.get("WS_JUDGE_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    cli = boto3.client("bedrock-runtime", region_name=region,
                       config=Config(retries={"total_max_attempts": 2, "mode": "standard"},
                                     connect_timeout=10, read_timeout=30))
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "user", "content": "Say OK."}]}
    try:
        r = cli.invoke_model(modelId=model, body=json.dumps(body),
                             contentType="application/json", accept="application/json")
    except Exception as e:
        if "temperature" in str(e).lower():
            body.pop("temperature", None)
            r = cli.invoke_model(modelId=model, body=json.dumps(body),
                                 contentType="application/json", accept="application/json")
        else:
            raise
    out = json.loads(r["body"].read())
    txt = "".join(b.get("text", "") for b in out.get("content", []))
    print(f"[ws judge self-test] Bedrock {model} @ {region} OK -> {txt!r}")
except Exception as e:
    print(f"[ws judge self-test] FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF

# Disaggregated vs colocate memory budgets (same logic as the tau/math _mns scripts).
# ROLLOUT_NUM_GPUS is exported by the Greenland bootstrap (0 / unset = colocate).
if [ "${ROLLOUT_NUM_GPUS:-0}" -gt 0 ]; then
    # File-tool trajectories are long multi-turn (read_file dumps inflate context),
    # like multi-turn tau — keep tau's 20480 (env-overridable). OOM-safe via
    # --log-probs-chunk-size (the entropy step's [T,V] doubling is bounded to [chunk,V]).
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-20480}
    SGLANG_MEM_FRACTION=0.85
else
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-9216}
    SGLANG_MEM_FRACTION=0.7
fi

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base/
   --ref-load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base_torch_dist/
   --load ${MODEL_ROOT}/Qwen3.5/Qwen3.5-4B-Base_torch_dist/
   --save ${MODEL_ROOT}/AECE/Qwen3.5-4B-Base-WS-async/
   # Save every 5 steps; permanently RETAIN every 20th (--save-retain-interval
   # must be a multiple of --save-interval, validated by Megatron). --save-retain-interval
   # is Megatron's native arg (do NOT re-add it in slime/utils/arguments.py — that
   # dup crashes parse_args; see memory slime-save-retain-interval-conflict).
   --save-interval 5
   --save-retain-interval 20
)

# Save every rollout's samples (.pt) for offline inspection / SFT distillation.
ROLLOUT_DEBUG_DIR="${MODEL_ROOT}/AECE/ws_rl/${WANDB_GROUP}"

ROLLOUT_ARGS=(
   --prompt-data "${WS_DATA_DIR}/ws_${WS_SPLIT}_train_tasks.jsonl"
   --input-key index
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt 16
   # Up from tau's 2048: file contents inflate the per-turn context vs dialogue.
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   # Dense [0,1] judge reward -> groups where all 16 samples score the same fraction
   # (e.g. all 0) carry no GRPO signal; this filter drops them.
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
   --log-passrate
   --save-debug-rollout-data "${ROLLOUT_DEBUG_DIR}/rollout_{rollout_id}.pt"
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data ws-dev "${WS_DATA_DIR}/ws_${WS_SPLIT}_dev_tasks.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-k 1
)

PERF_ARGS=(
   # TP=2 shards the [T, V=248320] fp32 logits head so the multi-turn entropy step
   # doesn't OOM. TP ceiling is 4 here (--num-query-groups 4 must be divisible by TP).
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   # Chunk the log-prob/entropy compute along the TOKEN dim so the peak buffer is
   # [chunk, V] (~1GiB) instead of a full [T, V] doubling (~21GiB). Numerically
   # identical (per-token values are independent). Critical for long multi-turn
   # trajectories (see memory tau-empty-microbatch-maxtokens).
   --log-probs-chunk-size 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   # Light KL leash to the ref (Qwen3.5-4B-Base) — prevents the policy-collapse the
   # tau run hit at coef 0. Tunable via env (try 0.005–0.02).
   --kl-loss-coef ${WS_KL_LOSS_COEF:-0.01}
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

# ── Router policy / session affinity ──
# consistent_hashing makes all turns of a trajectory reuse one worker's prefix
# cache (cuts repeated prefill of the growing history). Default cache_aware.
export WS_ROUTER_POLICY="${WS_ROUTER_POLICY:-cache_aware}"

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
   --sglang-server-concurrency 1024
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-disable-custom-all-reduce
   --router-policy ${WS_ROUTER_POLICY}
)

# ── MTP / EAGLE speculative decoding (single-model; goes straight into the global
# SGLANG_ARGS since there is no user-sim YAML to scope around). Qwen3.5 ships a
# built-in MTP head; EAGLE uses it as the draft. Training-LOSSLESS — only speeds
# generation. Toggle with WS_SPEC_DECODING=off. Setting the global arg also enables
# the spec_accept_rate / spec_accept_length wandb metrics.
WS_SPEC_DECODING="${WS_SPEC_DECODING:-eagle}"
SPEC_NUM_STEPS=${SPEC_NUM_STEPS:-3}
SPEC_EAGLE_TOPK=${SPEC_EAGLE_TOPK:-1}
SPEC_NUM_DRAFT_TOKENS=${SPEC_NUM_DRAFT_TOKENS:-4}
MAMBA_SCHED_STRATEGY="${MAMBA_SCHED_STRATEGY:-extra_buffer}"
if [ "${WS_SPEC_DECODING}" = "eagle" ]; then
   SGLANG_ARGS+=(
      --sglang-speculative-algorithm EAGLE
      --sglang-speculative-num-steps ${SPEC_NUM_STEPS}
      --sglang-speculative-eagle-topk ${SPEC_EAGLE_TOPK}
      --sglang-speculative-num-draft-tokens ${SPEC_NUM_DRAFT_TOKENS}
      --sglang-mamba-scheduler-strategy ${MAMBA_SCHED_STRATEGY}
   )
   echo "[ws] EAGLE speculative decoding ENABLED for the actor (steps=${SPEC_NUM_STEPS}, topk=${SPEC_EAGLE_TOPK}, draft_tokens=${SPEC_NUM_DRAFT_TOKENS})"
else
   echo "[ws] EAGLE speculative decoding DISABLED (WS_SPEC_DECODING=${WS_SPEC_DECODING}) — plain rollout"
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # dropout=0 so bias-dropout fusion is a no-op; with --use-dynamic-batch-size the
   # @jit_fuser recompiles every microbatch and hits recompile_limit -> hang.
   # Disable to take the pure-Python path (matches the base/tau _mns scripts).
   --no-bias-dropout-fusion
)

CUSTOM_ARGS=(
   # multi-turn file-editing rollout that drives the Workspace-Bench env + judge
   --custom-generate-function-path generate_with_ws.generate
   # TIS-related args (recommended to enable when using --use-tis)
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-0}
# workspace-bench example modules live in SCRIPT_DIR; they import the vendored
# ws_bench package (also under SCRIPT_DIR) and slime. Put SCRIPT_DIR on the path.
export PYTHONPATH="${SLIME_DIR:-${ROOT_DIR}/slime}:${MEGATRON_DIR:-${ROOT_DIR}/Megatron-LM}:${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"

# ── Prepare Workspace-Bench task data ──
# prepare_ws_data.py downloads the HF dataset + workspaces, normalizes each task's
# file_system to its *_raw workspace dir, and writes ws_<split>_{train,dev}_tasks.jsonl
# ({"index": i}) + task_index_map.json + tasks/ + workspaces/ under WS_DATA_DIR.
# Primary path: pre-staged on S3 (pull with --stage-data workspace-bench/). Only the
# MAIN node reaches here (workers block in `ray start --block`), and train.py loads
# --prompt-data on the head, so this runs once. Idempotent: skip if the train jsonl
# is already present (e.g. staged from S3).
mkdir -p "${WS_DATA_DIR}"
if [ -s "${WS_DATA_DIR}/ws_${WS_SPLIT}_${WS_TASK_SPLIT}_tasks.jsonl" ]; then
    echo "[ws data] ${WS_DATA_DIR}/ws_${WS_SPLIT}_${WS_TASK_SPLIT}_tasks.jsonl already present; skipping download"
else
    echo "[ws data] preparing Workspace-Bench (${WS_SPLIT}) -> ${WS_DATA_DIR}"
    ( cd "${SCRIPT_DIR}" && python3 prepare_ws_data.py --split "${WS_SPLIT}" --out "${WS_DATA_DIR}" --skip-upload )
fi
ls -l "${WS_DATA_DIR}"/ws_*_tasks.jsonl "${WS_DATA_DIR}"/task_index_map.json

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
# Tolerates long AWS Batch child-node boot-skew (180 x 10s = 30 min) and prints
# VARYING progress every iteration so the Greenland stuck-job detector doesn't
# kill the head while it waits.
EXPECTED_GPUS=$(( ACTOR_NUM_NODES * NUM_GPUS + ROLLOUT_NUM_GPUS ))
if [ "$EXPECTED_GPUS" -gt "$NUM_GPUS" ]; then
    echo "Waiting for ${EXPECTED_GPUS} GPUs (train $(( ACTOR_NUM_NODES * NUM_GPUS )) + rollout ${ROLLOUT_NUM_GPUS}) to register with Ray (head=${MASTER_ADDR}, up to 30 min for child-node boot-skew)..."
    WAIT_OK=0
    WAIT_START=$(date +%s)
    for i in $(seq 1 180); do
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
        echo "       Ray cluster — most likely control-plane port 6379 is blocked across subnets,"
        echo "       or the child node never booted. Check the WORKER node's CloudWatch stream."
        ray stop --force 2>/dev/null || true
        exit 1
    fi
    echo "All ${EXPECTED_GPUS} GPUs registered in $(( $(date +%s) - WAIT_START ))s; proceeding to submit."
fi

# Multi-node EFA/socket env. Actors on worker nodes inherit the JOB runtime-env,
# so socket-iface pins + EFA must live here.
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

# Every rollout actor (on every node) runs the rubric judge, so EVERY actor needs
# the WS_* knobs AND the AWS creds for the judge's boto3 chain. Inject into the Ray
# job runtime-env so worker-node actors inherit them (the worker shell env does NOT
# propagate to job tasks).
WS_ENV_JSON="
    \"WS_ENV\": \"${WS_ENV}\",
    \"WS_SPLIT\": \"${WS_SPLIT}\",
    \"WS_TASK_SPLIT\": \"${WS_TASK_SPLIT}\",
    \"WS_TOOL_PARSER\": \"${WS_TOOL_PARSER}\",
    \"WS_ENABLE_THINKING\": \"${WS_ENABLE_THINKING}\",
    \"WS_STRIP_HISTORICAL_THINK\": \"${WS_STRIP_HISTORICAL_THINK}\",
    \"WS_ENV_THREAD_WORKERS\": \"${WS_ENV_THREAD_WORKERS}\",
    \"WS_MAX_STEPS\": \"${WS_MAX_STEPS}\",
    \"WS_READ_CHAR_CAP\": \"${WS_READ_CHAR_CAP}\",
    \"WS_DATA_DIR\": \"${WS_DATA_DIR}\",
    \"WS_ROLLOUT_ROOT\": \"${WS_ROLLOUT_ROOT}\",
    \"WS_ROUTER_POLICY\": \"${WS_ROUTER_POLICY}\",
    \"WS_JUDGE_MODEL_ID\": \"${WS_JUDGE_MODEL_ID}\",
    \"WS_BEDROCK_REGION\": \"${WS_BEDROCK_REGION}\",
    \"WS_JUDGE_MAX_TOKENS\": \"${WS_JUDGE_MAX_TOKENS}\",
    \"WS_JUDGE_FILE_CHARS\": \"${WS_JUDGE_FILE_CHARS}\",
    \"AWS_CONFIG_FILE\": \"${AWS_CONFIG_FILE}\",
    \"AWS_DEFAULT_REGION\": \"${AWS_DEFAULT_REGION}\",
    \"AWS_REGION\": \"${AWS_DEFAULT_REGION}\""
if [ -n "${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-}" ]; then
    WS_ENV_JSON="${WS_ENV_JSON},
    \"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI\": \"${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI}\""
fi
if [ -n "${WS_JUDGE_PROFILE:-}" ]; then
    WS_ENV_JSON="${WS_ENV_JSON},
    \"WS_JUDGE_PROFILE\": \"${WS_JUDGE_PROFILE}\""
fi

# FI_EFA_FORK_SAFE=1 is UNCONDITIONAL (not gated on ACTOR_NUM_NODES>1): the EFA
# image bakes FI_PROVIDER=efa into its global ENV, so every process initializes
# the EFA provider regardless of node count, and Megatron's async checkpoint save
# fork()s — EFA SIGABRTs on the first save unless fork-safe is set.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"FI_EFA_FORK_SAFE\": \"1\",
    ${WS_ENV_JSON}${MULTINODE_ENV}
  }
}"

# ASYNC resource args: train_async.py REQUIRES disaggregated.
RESOURCE_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${NUM_GPUS}
   --num-gpus-per-node ${NUM_GPUS}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
)
echo "Disaggregated (async): actor=$(( ACTOR_NUM_NODES * NUM_GPUS )) GPU, rollout=${ROLLOUT_NUM_GPUS} GPU"

# train_async.py overlaps rollout(rollout_id+1) with train(rollout_id) so both GPU
# pools stay busy. --update-weights-interval 1 = update every step (closest to on-policy).
ASYNC_ARGS=(
   --update-weights-interval ${UPDATE_WEIGHTS_INTERVAL:-1}
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   ${RESOURCE_ARGS[@]} \
   ${ASYNC_ARGS[@]} \
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
