import torch

path = "/data2/whx/agentic_rl_search_log/rollout_30.pt"
d = torch.load(path, weights_only=False)

print("rollout_id:", d["rollout_id"])
print("samples 数量:", len(d["samples"]))

# # 看前 2 个 sample 的主要字段
# for i, s in enumerate(d["samples"][:10]):
#     print("\n--- sample", i, "---")
#     print("prompt :", (s.get("prompt") or "")[:], "...")
#     print("response :", (s.get("response") or "")[:], "...")
    # print("reward:", s.get("reward"))
    # print("response_length:", s.get("response_length"))

# 统计有多少个reward大于0的序列
count_positive_reward = 0
positive_reward_samples = []
for s in d["samples"]:
    if s.get("reward", 0) > 0:
        print(s.get("response"))
        count_positive_reward += 1
print(f"reward > 0 的序列数量: {count_positive_reward} / {len(d['samples'])}")