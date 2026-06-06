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
