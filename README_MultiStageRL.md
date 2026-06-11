## Multi-stage RL

### 8B Version
1. Math Reasoning
```bash
bash examples/maths_reasoning/run-qwen3-8B-base.sh
bash examples/maths_reasoning/resume-qwen3-8B-base.sh
```
Model is saved at `/xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math`

2. Agentic Search

2.1 Launch Retriever
The database is under `/xuanwu-tank/center/whx/Search_data`
```bash
cd Search-R1
conda activate retriever
bash Search-R1/retrieval_launch.sh
```

2.2 RFT
SFT data is saved at `/xuanwu-tank/center/whx/MultiStageRL/rollout_only_search_Qwen3-32B`
```bash
# 1.
bash examples/search-r1/run_qwen3_8b_rollout_only.sh 

# 2. 
python examples/search-r1/filter_rollout_data.py \
    --input-dir /xuanwu-tank/center/whx/MultiStageRL/rollout_only_search_Qwen3-8B \
    --output-dir /xuanwu-tank/center/whx/MultiStageRL/rollout_only_search_Qwen3-8B/filtered \
    --min-reward 0.6 --batch-size 64

# 3. 
bash examples/search-r1/run_qwen3_8b_sft.sh
```

2.3 RL
```bash
bash examples/search-r1/run_qwen3_8b_seq_gspo_sft.sh
```


3. Agentic Custom Service

3.1 Serve a user model

3.2 RFT
SFT data is saved at `/xuanwu-tank/center/whx/MultiStageRL/rollout_only_tau_Qwen3-32B`
```bash
bash examples/tau-bench/run_qwen3_8B_rollout_only.sh 

bash examples/tau-bench/run_qwen3_8B_sft.sh 
```
3.3 RL
```bash
bash examples/tau-bench/run_qwen3_8B.sh
```

### Qwen3.5-9B version
1. Run the docker
```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /xuanwu-tank/center/whx:/xuanwu-tank/center/whx \
  -it slimerl/slime:latest /bin/bash
```
Install the env
```bash
# 路径可根据实际情况调整
cd /root/slime
git pull
pip uninstall -y nvidia-cudnn-cu12 nvidia-cudnn-cu11 nvidia-cudnn 2>/dev/null || true
pip install -e . --no-deps
```

Quit and save the docker
```bash
# Find docker container ID
docker ps
docker commit xxx whx/MSRL:latest
```
<!-- export  CUDA_VISIBLE_DEVICES=6,7
sglang serve \
  --model-path /xuanwu-tank/center/whx/Qwen3.5/Qwen3.5-9B-Base \
  --dtype bfloat16 \
  --tp 1 \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code -->

2. VLM Reasoning

2.1 Download the data
```bash
hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed --local-dir /xuanwu-tank/center/whx/geo3k_imgurl_processed
```

2.2


## OPD

1. Deploy the teacher on scai7
```bash
# --model-path /xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math/hf_iter_0000300 \
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path /xuanwu-tank/center/whx/MultiStageRL/Qwen3-8B-Base-Math-SeaSFT-Search/hf_iter_0000420 \
  --host 0.0.0.0 --port 30012 \
  --tp 1 --mem-fraction-static 0.9 \
  --reasoning-parser qwen3 --tool-call-parser qwen
```