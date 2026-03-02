import torch

path = "/data2/whx/Qwen3-30B-A3B_base_math/rollout_debug/rollout_eval_0.pt"
d = torch.load(path, weights_only=False)

print("rollout_id:", d["rollout_id"])
print("samples 数量:", len(d["samples"]))

# 看前 2 个 sample 的主要字段
for i, s in enumerate(d["samples"][:2]):
    print("\n--- sample", i, "---")
    print("prompt :", (s.get("prompt") or "")[:], "...")
    print("response :", (s.get("response") or "")[:], "...")
    print("reward:", s.get("reward"))
    print("response_length:", s.get("response_length"))