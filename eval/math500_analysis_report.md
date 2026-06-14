# math500 详细分析报告（base / base_math / final_search / tau2）

脚本：
- [tools/eval_analysis/classify_math_failures.py](../tools/eval_analysis/classify_math_failures.py)
- [tools/eval_analysis/breakdown_math_eval.py](../tools/eval_analysis/breakdown_math_eval.py)
- [tools/eval_analysis/regrade_math_jsonl.py](../tools/eval_analysis/regrade_math_jsonl.py)

CSV 导出：`/tmp/math500_per_doc_4ckpt.csv`（500 行，每个 ckpt 一列 0/1 + pattern 列）

> tau2 的 jsonl 是用 patched utils.py（含 sympy fallback）跑出来的，原始 `exact_match` 已经是 sympy 等价判定后的分数，**不需要再 regrade**。

---

## 1. 总体准确率（4 ckpt）

| | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|
| harness 原始 | 0.7960 | 0.8480 | 0.8520 | **0.8600** | +6.40pp |
| + 文本 hidden_correct rescue | **0.8040** | 0.8480 | 0.8540 | 0.8600 | +5.60pp |
| 救起来的题数 | +4 solid + 2 fragile | 0 | +1 solid | 0 | — |

**关键发现**：tau2 是 4 个 ckpt 里**分数最高**的（0.8600，比 final_search 高 0.6pp）。tau2 没有 hidden_correct_text 救起来的（boxed 格式覆盖率高，bare 文本回答几乎没漏）。

---

## 2. Layer 1/2/3 失败审计（4 ckpt）

### Layer 1 — 框内判决

| | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| `agree_correct` (harness ✓ slime ✓) | 398 | 424 | 426 | **430** |
| `hidden_correct_box` | 0 | 0 | 0 | 0 |
| `boxed_demoted` | 0 | 0 | 0 | 0 |
| `agree_wrong` (有 \boxed 但答错) | 77 | 71 | 68 | **65** |
| `no_boxed` | 25 | 5 | 6 | **5** |

→ tau2 把 `agree_wrong` (有框但错) 进一步降到 65；`no_boxed` 跟 base_math 持平 (5)。

### Layer 2 — no_boxed 内部

| | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| no_boxed 总数 | 25 | 5 | 6 | 5 |
| `hidden_correct_text` solid | 4 | 0 | 1 | **0** |
| `hidden_correct_text` fragile | 2 | 0 | 0 | 0 |
| `hidden_failure` (prose 承诺错答案) | 2 | 1 | 0 | **2** |
| no commitment | 17 | 4 | 5 | 3 |

→ tau2 的 5 个 no_boxed 里 2 个是 hidden_failure（prose 给出错答案），3 个是没收尾——格式问题已经清得很干净了。

### Layer 3 — 失败子类

| 子类 | base | base_math | final_search | **tau2** |
|---|---:|---:|---:|---:|
| `wrong_answer` (有框、确实错) | 77 | 71 | 68 | **65** |
| `hidden_failure` | 2 | 1 | 0 | 2 |
| `fragile_text_match` | 2 | 0 | 0 | 0 |
| `reach_max_function_call` | 6 | 0 | 0 | **0** |
| `python_block_unfinished` | 1 | 0 | 0 | 0 |
| `repetition_loop` | 3 | 0 | 3 | **0** |
| `truncated_max_tokens` | 3 | 2 | 1 | **0** |
| `medium_unconverged` | 3 | 2 | 1 | 2 |
| `gave_up_short` | 1 | 0 | 0 | 1 |

→ tau2 上 `repetition_loop`、`truncated_max_tokens`、`reach_max_function_call` **全部归零**——格式 / 训练遗留 bug 已经基本扫清。

---

## 3. 按难度档（Level）

| Level | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 43 | 0.9302 | 0.9302 | 0.9070 | **0.9535** | +2.3pp |
| L2 | 90 | 0.9333 | 0.9444 | 0.9333 | **0.9556** | +2.2pp |
| L3 | 105 | 0.9524 | 0.9619 | 0.9429 | 0.9524 | ±0 |
| L4 | 128 | 0.7891 | 0.8516 | **0.8594** | 0.8516 | +6.3pp |
| **L5** | 134 | 0.5746 | 0.6642 | **0.7090** | 0.7015 | **+12.7pp** |

**观察**：
- tau2 在 **L1/L2 都创新高**（base 之后没有任何 ckpt 能超）—— 简单题的小波动被收拾干净
- L5 tau2 略低于 final_search（94 vs 95，差 1 题），但仍比 base_math 高
- L4 tau2 与 base_math 持平（109 vs 109），低于 final_search 110

---

## 4. 按题型（Subject）

| 题型 (n) | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|
| Algebra (124) | 0.9113 | 0.9677 | **0.9919** | 0.9677 | +5.6pp |
| Counting & Probability (38) | 0.7895 | **0.8947** | 0.8421 | 0.8421 | +5.3pp |
| **Geometry (41)** | 0.6585 | 0.6585 | 0.6829 | **0.7073** | **+4.9pp** ⭐ |
| Intermediate Algebra (97) | 0.6186 | 0.6804 | **0.7113** | 0.7113 | +9.3pp |
| **Number Theory (62)** | 0.9677 | 0.9516 | 0.9194 | **0.9677** | **±0** ⭐（恢复！）|
| **Prealgebra (82)** | 0.9024 | 0.8780 | 0.9024 | **0.9390** | **+3.7pp** ⭐ |
| Precalculus (56) | 0.6786 | **0.8214** | 0.7857 | 0.7679 | +8.9pp |

**tau2 的三大独特贡献**（⭐ 标记）：

1. **Number Theory 完全恢复到 base 水平**：之前 base_math (-1) → final_search (-3 累计) 持续退化，tau2 直接把丢的 3 题全部修回来 → 60/62（与 base 完全一致）
2. **Geometry 创新高**：base/base_math 卡在 0.6585，final_search 微涨到 0.6829，tau2 推到 **0.7073**（4 ckpt 第一次超过 0.70）
3. **Prealgebra 创新高**：之前最高 0.9024（base/final_search），tau2 拉到 **0.9390**

**tau2 的回退**：
- Algebra：丢了 final_search 拿到的"满分接近线" 0.9919（123/124），退回 0.9677（120/124，与 base_math 持平）
- Precalculus：连续两轮回退（0.8214 → 0.7857 → 0.7679）

---

## 5. 三对相邻 transition + 端到端

### base → base_math（math RL）

```
fixed=39, broken=17, net=+22
fixed by level:   L1:1, L2:3, L3:3, L4:14, L5:18
broken by level:  L1:1, L2:2, L3:2, L4:6, L5:6
```

### base_math → final_search（search 训练）

```
fixed=19, broken=16, net=+3
fixed by level:   L3:2, L4:4, L5:13     （13/19 在 L5）
broken by level:  L1:1, L2:1, L3:4, L4:3, L5:7
```

### **final_search → tau2（最新阶段）**

```
fixed=20, broken=17, net=+3
fixed by level:   L1:2, L2:3, L3:4, L4:4, L5:7
broken by level:  L2:1, L3:3, L4:5, L5:8
```

### base → tau2（端到端）

```
fixed=42, broken=14, net=+28
fixed by level:   L1:1, L2:4, L3:3, L4:14, L5:20
broken by level:  L2:2, L3:3, L4:6, L5:3
```

→ 端到端 fixed:broken ≈ 3:1，比之前 base→final_search 的 40:15 还略好。

---

## 6. Fixed/Broken 矩阵（level × subject，关注 final_search → tau2）

### final_search → tau2 FIXED (20 道)

| Level | Algebra | C&P | Geom | IntAlg | NT | Prealg | Precalc | **行计** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | · | · | 1 | · | 1 | · | · | **2** |
| L2 | · | · | · | 1 | · | 1 | 1 | **3** |
| L3 | · | · | 1 | · | 2 | 1 | · | **4** |
| L4 | · | · | · | 3 | 1 | · | · | **4** |
| L5 | · | · | 1 | 2 | 1 | 2 | 1 | **7** |
| **列计** | **0** | **0** | **3** | **6** | **5** | **4** | **2** | **20** |

### final_search → tau2 BROKEN (17 道)

| Level | Algebra | C&P | Geom | IntAlg | NT | Prealg | Precalc | **行计** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | · | · | · | 1 | · | · | · | **1** |
| L3 | 1 | · | · | · | · | · | 2 | **3** |
| L4 | 1 | · | · | 1 | 1 | 1 | 1 | **5** |
| L5 | 1 | · | 2 | 4 | 1 | · | · | **8** |
| **列计** | **3** | **0** | **2** | **6** | **2** | **1** | **3** | **17** |

**关键观察**：
- **Number Theory 5 fixed / 2 broken = net +3**：这是 tau2 在 NT 上的核心收益（从 final_search 57/62 一路修到 60/62）
- **Algebra 0 fixed / 3 broken**：tau2 在 Algebra 上反向退步，主要在 L3-L5（之前 final_search 的 L5 Algebra 是满分 30/30，tau2 退回 29/30）
- **L5 IntAlg 2 fixed / 4 broken**：tau2 在 GPQA 训练目标的核心硬区上反而略退，是个值得注意的副作用

### base → tau2 FIXED (42 道)

| Level | Algebra | C&P | Geom | IntAlg | NT | Prealg | Precalc | **行计** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 1 | · | · | · | · | · | · | **1** |
| L2 | 1 | · | · | 1 | · | 1 | 1 | **4** |
| L3 | · | · | 1 | · | · | · | 2 | **3** |
| L4 | 1 | 1 | 1 | 5 | 2 | · | 4 | **14** |
| L5 | 5 | 2 | 2 | 6 | · | 3 | 2 | **20** |
| **列计** | **8** | **3** | **4** | **12** | **2** | **4** | **9** | **42** |

### base → tau2 BROKEN (14 道)

| Level | Algebra | C&P | Geom | IntAlg | NT | Prealg | Precalc | **行计** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 | · | · | 1 | 1 | · | · | · | **2** |
| L3 | 1 | · | · | · | · | · | 2 | **3** |
| L4 | · | · | 1 | 1 | 1 | 1 | 2 | **6** |
| L5 | · | 1 | · | 1 | 1 | · | · | **3** |
| **列计** | **1** | **1** | **2** | **3** | **2** | **1** | **4** | **14** |

→ 端到端净 +28 题，集中在 L4-L5（+11 + +17 = +28）；broken 很分散，没有单格 > 2 道。

---

## 7. 4-ckpt Pattern 统计

| Pattern (base → bm → fs → tau2) | 数量 | 含义 |
|---|---:|---|
| `stable` (✓✓✓✓ + ✗✗✗✗) | **413** (82.6%) | 372 全对 + 41 全错 |
| `fixed_at_bm` (✗✓✓✓) | 29 | math RL 修好后稳定保持到 tau2 |
| `fixed_at_tau2` (✗✗✗✓) | 8 | **tau2 才修出来的新增题** |
| `fixed_at_fs` (✗✗✓✓) | 2 | search 修好且 tau2 守住 |
| `broken_at_tau2` (✓✓✓✗) | 6 | **tau2 永久打破的题** |
| `broken_at_bm` (✓✗✗✗) | 5 | math RL 永久破坏 |
| `broken_at_fs` (✓✓✗✗) | 1 | search 破坏后 tau2 没修回 |
| `volatile_*` (波动型) | 36 | 含重新洗牌 / 反复 flip 的题 |

**Volatile 细分（4-ckpt bitstring）**：
- `volatile_1011` (✓✗✓✓): 7 — base 对、bm 破坏、fs 恢复并保持
- `volatile_0010` (✗✗✓✗): 8 — fs 短暂修好但 tau2 又破坏
- `volatile_0100` (✗✓✗✗): 6 — bm 短暂修好后 fs/tau2 都没保住
- `volatile_1101` (✓✓✗✓): 6 — fs 破坏后 tau2 修回（含 NT 那 3 道）
- `volatile_0101`/`0110`/`1001`/`1010` 等小桶

**重要解读**：
- 8 个 `fixed_at_tau2` + 6 个 `volatile_1101` 是 tau2 真正的"新增贡献"（共 14 题）
- 6 个 `broken_at_tau2` 是 tau2 独有的退步（其中 4 道在 L4-L5 的代数类）

---

## 8. Persistent 状态（4 ckpt 全错）

```
total: 500
persistent ✓ (4 ckpt 全对): 372  (74.4%)
persistent ✗ (4 ckpt 全错): 41   (8.2%)   ← 比 3-ckpt 的 49 减少 8 道！
flapping (变动):           87   (17.4%)
```

**persistent_wrong 减少 8 道** —— tau2 把之前三 ckpt 都做不出来的硬骨头里又拿下了 8 道。

按学科分布（41 道）：
- Intermediate Algebra: 19（之前 22，少 3）
- Geometry: 7（之前 8，少 1）
- Precalculus: 7（与之前同）
- Counting & Probability: 4（与之前同）
- Prealgebra: 3（之前 5，少 2）
- Algebra: 1（与之前同）
- **Number Theory: 0（之前 1，全部破除）** ⭐

按难度（41 道）：L5: 28, L4: 8, L3: 1, L2: 2, L1: 2 —— 仍然 88% 集中在 L4-L5。

---

## 9. 与之前 ckpt 的核心结论变化

| 维度 | 3-ckpt 结论 | 加上 tau2 后 |
|---|---|---|
| 最弱学科 | Geometry 持续低位 (0.66-0.68) | **Geometry 终于在 tau2 突破 0.70** |
| Number Theory 趋势 | 持续退化 (-4.8pp，叠加性) | **tau2 完全恢复到 base 水平** |
| L5 增长前沿 | L5 +13.4pp 是主战场 | tau2 略退一题 (95→94)，主战场转向 L1-L2 收尾 |
| 训练边界 | L5 IntAlg 18 道 persistent_wrong | tau2 把这一格修了几道，整体 persistent_wrong 下降 8 道 |
| Algebra 满分线 | final_search 拿到 30/30 L5 | **tau2 退回 29/30** |

---

## 10. 跨数据集横向对比（4 ckpt）

| 指标 | math500 | MMLU | MMLU-Pro | GPQA |
|---|---:|---:|---:|---:|
| 总题数 | 500 | 14042 | 12032 | 198 |
| base 起点 | 0.7960 | 0.6699 | 0.5677 | 0.3788 |
| base_math | 0.8480 | 0.7554 | 0.6516 | 0.4697 |
| final_search | 0.8540 | 0.7945 | 0.6971 | 0.4949 |
| **tau2** | **0.8600** | (跑中) | (跑中) | (跑完*) |
| base→tau2 Δ | +6.4pp | — | — | — |
| persistent_wrong | 41 (8.2%) | — | — | — |

*tau2 GPQA 数据已经跑完，等到大任务（mmlu/mmlu_pro）也跑完后我再做完整 4-ckpt 对比报告。

---

## 11. 核心结论

1. **tau2 是 math500 上最佳 ckpt**（0.8600），比 final_search 净 +0.6pp，端到端比 base +6.4pp
2. **tau2 的两个独特修复**：
   - Number Theory 完全恢复（前两轮训练累计丢 3 题，tau2 全部修回）
   - Geometry 突破 0.70 天花板（之前 4 ckpt 中只有 tau2 做到）
3. **tau2 的代价是 Algebra L5 满分丢失**：final_search 那 30/30 退回 29/30
4. **格式相关失败几乎清零**：tau2 上 `reach_max_function_call`、`repetition_loop`、`truncated_max_tokens` 全 0；只剩 5 道 no_boxed（其中 2 道 hidden_failure，3 道未收尾）
5. **persistent_wrong 从 49 降到 41**：tau2 拿下了 8 道之前三 ckpt 都搞不定的硬题，且 L4-L5 仍占 88%（训练边界没本质变化）
6. **Pattern 统计透露的训练特征**：
   - `fixed_at_tau2` 8 道 + `volatile_1101` 6 道 = 14 道 tau2 真新增
   - `broken_at_tau2` 6 道 + `volatile_0010` 8 道 = 14 道 tau2 新破坏
   - 净增 = +0 但实质是"换了一批题"，不是均匀提升
