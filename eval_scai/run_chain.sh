#!/bin/bash
# Sequential evaluation chain: one checkpoint at a time, full GPU per model.
# (Parallel-on-one-GPU oversubscribed compute: 3 servers x 16 conns x 32k-token
# CoT made single requests exceed 1h and trip APITimeoutError. Sequential gives
# each model ~3x lower latency for the same total compute.)
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

GPU="${GPU:-3}"
export EVAL_GPUS="${GPU}" EVAL_PORT=8400 EVAL_MEM_FRACTION=0.85
export INSPECT_MAX_CONNECTIONS=24 INSPECT_TIMEOUT=7200

run_one () {
    local ckpt="$1" name="$2"; shift 2
    local benches="$*"
    echo "############ [$(date +%F' '%T)] ${name}: ${benches} ############"
    CKPT="${ckpt}" CKPT_NAME="${name}" bash "${SCRIPT_DIR}/run_all.sh" ${benches}
    # tear down this model's server so the next one gets the GPU
    local spid
    spid=$(cat "${SCRIPT_DIR}/logs/${name}/eval_server.pid" 2>/dev/null || true)
    [ -n "${spid}" ] && kill -9 "${spid}" 2>/dev/null
    sleep 10
}

# base: gpqa, aime, ifbench already done (results kept) — run the rest
run_one /xuanwu-tank/center/whx/Qwen3-8B-Base Qwen3-8B-Base \
        ifeval search mmlu browsecomp tau2

run_one /xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300 Qwen3-8B-Base-Math \
        gpqa aime ifeval ifbench search mmlu browsecomp tau2

run_one /xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420 Qwen3-8B-Base-Math-SeaSFT-Search \
        gpqa aime ifeval ifbench search mmlu browsecomp tau2

echo "############ [$(date +%F' '%T)] CHAIN COMPLETE ############"
