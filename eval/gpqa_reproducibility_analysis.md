# GPQA 可重复性分析（dp=1 vs dp=2，base / base_math）

> **TL;DR**：dp 设置**不是**主要噪声源。**同一台 server、同一组 dp 设置、temp=0** 下连跑两次，准确率差就能到 ±7pp（base）/ ±2pp（base_math）。GPQA 198 题这么小的样本 + Qwen3-30B-A3B 这种 MoE 模型 + 并发推理（conc=18-35），单次实验的"准确率"对真实模型能力的估计有 **±5pp 的噪声地板**。

源数据：[/tmp/gpqa_dp_repro_analysis.txt](file:///tmp/gpqa_dp_repro_analysis.txt)

---

## 1. 准确率全表

### base（GPQA Diamond, 198 题）

| run | acc | n_correct |
|---|---:|---:|
| dp=2 run1（5/4 20:38）| 0.3232 | 64 |
| dp=2 run2（5/4 20:53）| **0.3939** | 78 |
| dp=1 run1（5/5 12:15）| 0.3434 | 68 |
| dp=1 run2（5/5 12:22）| **0.4091** | 81 |

range = **0.3232 ↔ 0.4091 = 17 道差距 = 8.6pp**

### base_math

| run | acc | n_correct |
|---|---:|---:|
| dp=2 run1（5/4 20:19）| **0.4899** | 97 |
| dp=1 run1（5/5 12:17）| 0.4545 | 90 |
| dp=1 run2（5/5 12:23）| 0.4697 | 93 |

range = **0.4545 ↔ 0.4899 = 7 道 = 3.5pp**

---

## 2. 关键洞察：**同 dp 内噪声 ≈ 跨 dp 噪声**

### base 的 6 对 pairwise flip 数

| pair | flip 数 | A→B  | B→A |
|---|---:|---:|---:|
| **dp2_run1 vs dp2_run2** ⭐同 dp | **46** | 16 | 30 |
| **dp1_run1 vs dp1_run2** ⭐同 dp | **45** | 16 | 29 |
| dp2_run1 vs dp1_run1 | 30 | 13 | 17 |
| dp2_run1 vs dp1_run2 | 55 | 19 | 36 |
| dp2_run2 vs dp1_run1 | 46 | 28 | 18 |
| dp2_run2 vs dp1_run2 | 27 | 12 | 15 |

→ **同 dp 内连跑两次会 flip 45-46 道**，跨 dp 反而有几对 flip 更少（27、30）。这意味着 dp 设置**不是** GPQA 噪声的主要来源。

### base_math 的 3 对

| pair | flip 数 |
|---|---:|
| dp2_run1 vs dp1_run1 | 21 |
| dp2_run1 vs dp1_run2 | 42 |
| **dp1_run1 vs dp1_run2** ⭐同 dp | **37** |

base_math 同 dp 内也 flip 37 道。

---

## 3. 字节级生成的"非确定性"——这是根因

> **同样的 prompt + temperature=0 + 同 dp 设置 → response 字节相等率只有 16-31%**

| pair | byte-identical | 解读 |
|---|---:|---|
| **dp2_run2 vs dp1_run2** | 62/198 (31%) | 跨 dp 反而最相似 |
| dp2_run1 vs dp1_run1 | 37/198 (19%) | |
| **dp1_run1 vs dp1_run2** ⭐同 dp | 36/198 (18%) | |
| **dp2_run1 vs dp2_run2** ⭐同 dp | 32/198 (16%) | |
| dp2_run2 vs dp1_run1 | 13/198 (7%) | |
| dp2_run1 vs dp1_run2 | 16/198 (8%) | |

→ **任意两次 GPQA 跑出来约 70-93% 的回答都不是字节相等的**。即使 temp=0 应该是 greedy decoding，实际 sample-by-sample 输出仍然在变。

### 为什么 temp=0 还会变？

Qwen3-30B-A3B 是 **MoE 模型**，在 sglang 这种并发批处理 server 上：
1. **请求路由顺序变化**：每次跑请求到达 server 的微秒级时序不同 → 进入哪个 decoding batch 不同 → 跟谁同 batch 不同
2. **MoE token routing**：同 batch 内每个 token 选哪些 expert 是 dynamic 的（top-k routing），batch 组成变了 → expert 负载变了 → reduction 顺序变了
3. **FP32 reduction 顺序**：attention 和 MoE expert 的 sum reduction 在 GPU 上是 non-associative 的，1+ε 之差累积 ~3000 token 后能让 logit 排序翻转
4. **Greedy 不救**：argmax 的两位最大 logit 差 ε 时，浮点扰动可以让胜出 token 翻面，然后路径分叉，后面整段输出都变

**dp 影响**：dp=2 比 dp=1 的批组合方式不同（请求分到 2 个 worker 各自批 vs 1 个 worker 全部批），所以差异略大但不主导。

---

## 4. 哪些题在 flap

### base 4-run pattern（top 10）

| dp2_run1 | dp2_run2 | dp1_run1 | dp1_run2 | count |
|:-:|:-:|:-:|:-:|---:|
| ✗ | ✗ | ✗ | ✗ | **87** （持续答错）|
| ✓ | ✓ | ✓ | ✓ | **38** （持续答对）|
| ✗ | ✓ | ✗ | ✓ | 16 |
| ✗ | ✗ | ✗ | ✓ | 8 |
| ✗ | ✓ | ✓ | ✓ | 8 |
| ✓ | ✗ | ✓ | ✗ | 7 |
| ✗ | ✓ | ✗ | ✗ | 6 |
| ✓ | ✗ | ✗ | ✗ | 6 |
| ✗ | ✗ | ✓ | ✗ | 5 |
| ✗ | ✗ | ✓ | ✓ | 4 |

→ 198 题里：
- **87 (44%) 是稳定答错**：这是真正的能力盲点，跟 dp / run 无关
- **38 (19%) 是稳定答对**：模型真正会的题
- **73 (37%) 是 flap**：在 4 次实验里至少反转过一次——这部分**完全是噪声**，对训练效果不应该归因

### base_math 3-run pattern

- 79 持续答错（40%）
- 69 持续答对（35%）
- **50 在 flap（25%）**

→ base_math 的 flap 比例（25%）比 base 的（37%）低，说明 **math RL 后模型对 GPQA 题的"信心"提升了**——更多题落在了 logit 安全区，不容易被批处理扰动改答案。

### Flap 的 domain 分布

每对 pair 的 flip 都集中在：
- **Chemistry** > **Physics** > Biology

例如 dp2_run1 vs dp2_run2 的 46 个 flip：Chemistry 24 / Physics 18 / Biology 4。这些都是模型本来就回答不太确定的 domain（Organic Chemistry 占 GPQA 36%，正是稳定盲点重灾区）。

---

## 5. Response length 也在变

| run | avg | median | p95 |
|---|---:|---:|---:|
| base dp2_run1 | 3147 | 1702 | 8728 |
| base dp2_run2 | 3718 | 1646 | 15692 |
| base dp1_run1 | 3832 | 1702 | 17252 |
| **base dp1_run2** | **4239** | 1663 | **23966** |

→ 同 ckpt 同 prompt 同 temp，**p95 长度从 8.7k 飙到 24k**。后期长尾 generation（重复、循环、未收尾）的发生频率随 batch 状态变化，进而影响最终是否能写出 `\boxed{X}`、是否被截断 → 跟"答错"在统计上强相关。

---

## 6. 为什么 dp=1 反而 acc 更高？

平均 acc：
- dp=2 两次平均：**0.3586**
- dp=1 两次平均：**0.3763**（高 +1.8pp）

不能武断说"dp=1 更准"——只有各 2 次实验，差异完全可以是采样噪声（标准差估计 ~3-4pp，差 1.8pp 不显著）。但有一个可能的 mechanism：

> dp=2 把请求分到 2 个 worker 各自批 → 每个 worker 的批次更大 → MoE expert 负载更挤 → fp 累积误差更大；dp=1 单 worker 反而批次小、reduction 更稳定？

但需要 ≥10 次实验才能可靠回答。

---

## 7. 实践建议

1. **GPQA 单次结果误差范围 ±5pp**：现有 4-ckpt 的 GPQA 报告里"tau2 vs final_search 差 -2pp（旧 prompt）"或者"tau2 vs final_search 差 +3pp（新 prompt）"这种**幅度<5pp 的差异不能可靠归因到训练效果**——很可能就是噪声。
2. **要可靠对比，至少跑 N=3 取均值**：以现有数据为例，base 4 次平均 0.3674，标准差 ≈0.038（4pp），所以 95% CI ≈ ±4pp。
3. **Domain 级别的趋势仍然可信**：Physics 上 base→tau2 +25-35pp，Quantum Mechanics +30-40pp 这种大幅度变化，远超噪声地板，是真正的训练效果。
4. **Subdomain 级别（n<20）的趋势要小心**：比如 Genetics 4 题，单次结果 ±25pp 完全可能是噪声。
5. **MMLU/MMLU-Pro 受影响小**：那些数据集 12k+ 题，CLT 下噪声 / sqrt(n) 摊薄到 <1pp。math500 (500 题) 也比 GPQA (198 题) 稳。
6. **如果非要追求确定性**：要么 num_concurrent=1（慢 18 倍），要么换非 MoE 模型，要么忍受噪声但跑多次。

---

## 8. 跟之前 4-ckpt GPQA 报告的关系

之前的 [gpqa_newprompt_analysis_report.md](gpqa_newprompt_analysis_report.md) 里我说：
- "**新 prompt 让 tau2 重夺 GPQA 第一**：0.4949（vs final_search 0.4646，+3.0pp）"
- "final_search→tau2 +6 道（净正）"

**新认知**：3pp / 6 道的差距已经在噪声范围内（base 4 次 range = 17 道 = 9pp），所以这个"tau2 比 final_search 好"的结论**实际上没有统计显著性**。需要每个 ckpt 跑 ≥3 次才能下这个判断。

不过报告里更大的 deltas 仍然可信：
- **base→tau2 +34 道（+17.2pp）**：远超噪声
- **Physics +30 道**（base→tau2）：远超噪声
- **Quantum Mechanics 0.480 → 0.840**：远超噪声

→ "tau2 vs final_search 谁更好" 这种小差距的判断不可靠；"训练后比训练前显著提升"这种大差距判断仍然可靠。

---

## 9. 最重要的一句话

**Qwen3-30B-A3B 的 MoE 推理在 sglang 并发服务下不是确定性的**——同一组 server、同一组 dp 设置、temp=0、同 prompt，198 题 GPQA 仍然有 18-37% 的题在 run 之间 flap，对应 ±5pp 的 acc 噪声。dp=1 vs dp=2 的差距大部分被这个噪声本身吃掉了。
