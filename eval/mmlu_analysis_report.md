# MMLU 详细分析报告（base / base_math / final_search / tau2）

脚本：
- [tools/eval_analysis/classify_mmlu_failures.py](../tools/eval_analysis/classify_mmlu_failures.py)
- [tools/eval_analysis/breakdown_mmlu_eval.py](../tools/eval_analysis/breakdown_mmlu_eval.py)
- [tools/eval_analysis/regrade_mmlu_jsonl.py](../tools/eval_analysis/regrade_mmlu_jsonl.py)
- [tools/eval_analysis/_mmlu_subjects.py](../tools/eval_analysis/_mmlu_subjects.py)（57 → 4 类映射）

CSV 导出：`/tmp/mmlu_per_doc_4ckpt.csv`（14042 行，每个 ckpt 一列 0/1 + pattern 列）

---

## 1. 总体准确率

| | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|
| extractor 原始 | 0.6699 | 0.7554 | 0.7945 | **0.8281** | **+15.82pp** |
| + aggressive rescue | 0.6705 | 0.7557 | 0.7968 | **0.8288** | +15.83pp |
| 救起来的题数 | +8 | +4 | +32 | +10 | — |

**关键发现**：tau2 是 4 ckpt 中最高（0.8281），比 final_search 净提升 **+3.36pp**。这是 tau2 在 MMLU 上最大的进步，也是所有 4 个数据集中 tau2 vs final_search 涨幅最大的一个。

---

## 2. Layer 1 — 提取方法分布（揭示 RL 训练对输出格式的塑造）

| 抽取模式 | base | base_math | final_search | **tau2** |
|---|---:|---:|---:|---:|
| `Answer: X` 强制格式 | 12534 (89.3%) | 12187 (86.8%) | 12535 (89.3%) | **12742 (90.7%)** |
| `the answer is X` | 1151 (8.2%) | 135 (1.0%) | 77 (0.5%) | **1157 (8.2%)** |
| `\boxed{X}` | 37 (0.3%) | **949 (6.8%)** | **1005 (7.2%)** | **39 (0.3%)** |
| `no_match` | 320 (2.3%) | **771 (5.5%)** | 425 (3.0%) | **104 (0.7%)** |

**核心观察**：
1. **tau2 几乎完全回归 base 风格**：
   - `Answer: X` 比例 90.7%，最高
   - `the answer is X` 1157，跟 base 完全一样（base 是 1151）
   - `\boxed{X}` 39，跟 base 几乎一样（base 37）—— 完全**消除了 math RL 带来的格式漂移**
2. **`no_match` 创新低**：104（0.7%），是之前最低值的 1/3 —— tau2 几乎完全清除了"输出找不到答案字母"这种格式问题
3. **base_math → tau2 的 boxed 用法变化**（949 → 1005 → 39）：math RL 强行塞了 boxed 格式，search 训练保持了，**tau2 又主动卸下 boxed 回归自然 MCQ 格式**

---

## 3. Layer 2 — no_match 内部审计

| | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| no_match 总数 | 320 | 771 | 425 | **104** |
| `hidden_correct_letter` | 8 | 4 | 32 | **10** |
| `hidden_failure` (wrong) | 4 | 12 | 19 | **1** |
| `no_letter_found` | 308 | 755 | 374 | **93** |

**关键发现**：
- tau2 的 no_match 里 `hidden_failure` 几乎清零（仅 1 道）—— 模型不再有"prose 给了错答案但没用规范格式"的样本
- aggressive rescue 救起来 10 道（占 tau2 no_match 的 9.6%）

---

## 4. Layer 3 — 失败子类

| 子类 | base | base_math | final_search | **tau2** |
|---|---:|---:|---:|---:|
| `extracted_wrong` | 4315 | 2664 | 2460 | **2310** | 真正推理错误 |
| `hidden_failure` | 4 | 12 | 19 | 1 |
| `gave_up_short` | 240 | **502** | 147 | **56** |
| `medium_unconverged` | 61 | **247** | 196 | **29** |
| `truncated_max_tokens` | 0 | 0 | 25 | 7 |
| `repetition_loop` | 3 | 6 | 5 | 1 |
| `python_block_unfinished` | 4 | 0 | 1 | 0 |

**关键观察**：
- **`extracted_wrong` 持续下降** (4315 → 2310)：真实推理错误从 30.7% → 16.5%，符合 acc 持续上升
- **`gave_up_short` 大幅减少**（502 → 56）：base_math 阶段 math RL 副作用造成的"短回答放弃"被 tau2 治好了
- **`medium_unconverged` 也大幅减少**（247 → 29）：中等长度无答案的样本几乎清空
- **格式相关失败几乎清零**：`repetition_loop` 1 道、`python_block` 0 道、`reach_max_function_call` 全程 0

---

## 5. 按 4 大类别（category）

| Category | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| **STEM** | 3153 | 0.7491 | 0.8325 | 0.8716 | **0.9052** | **+15.60pp** |
| **Humanities** | 4705 | 0.5877 | 0.6249 | 0.6761 | **0.7273** | **+13.96pp** |
| **Social Sciences** | 3077 | 0.7020 | 0.8281 | 0.8508 | **0.8772** | **+17.52pp** |
| **Other** | 3107 | 0.6823 | 0.8027 | 0.8400 | **0.8539** | **+17.15pp** |

**全部四个 category 都创新高**：
- STEM 突破 0.90 大关 (0.9052)
- Social Sciences 涨幅最大 +17.5pp
- Humanities 仍是最弱（绝对值 0.7273），但 +13.96pp 不算少
- Other 跟 Social 接近，都是 +17pp 量级

---

## 6. Top 15 受益学科（base → tau2）

| 学科 | category | base | tau2 | Δ |
|---|---|---:|---:|---:|
| human_sexuality | Social | 0.634 | 0.893 | **+0.260** |
| formal_logic | Humanities | 0.643 | 0.897 | +0.254 |
| college_biology | STEM | 0.701 | **0.951** | +0.250 |
| college_mathematics | STEM | 0.670 | 0.910 | +0.240 |
| college_medicine | Other | 0.613 | 0.850 | +0.237 |
| moral_scenarios | Humanities | 0.420 | 0.650 | **+0.230** ⭐ |
| business_ethics | Other | 0.560 | 0.790 | +0.230 |
| professional_medicine | Other | 0.673 | **0.897** | +0.224 |
| marketing | Other | 0.718 | 0.932 | +0.214 |
| prehistory | Humanities | 0.685 | 0.898 | +0.213 |
| clinical_knowledge | Other | 0.660 | 0.872 | +0.211 |
| medical_genetics | Other | 0.730 | 0.940 | +0.210 |
| professional_psychology | Social | 0.624 | 0.830 | +0.206 |
| high_school_physics | STEM | 0.709 | 0.914 | +0.205 |
| high_school_mathematics | STEM | 0.748 | 0.948 | +0.200 |

**moral_scenarios 标 ⭐**：MMLU 上最大、最难的子类（895 题）——base 0.420 → tau2 0.650，+23pp，是绝对收益最大的"关键拼图"。

**没有 base → tau2 净退步的学科**——MMLU 的端到端训练对每一类都纯收益。

---

## 7. Fixed / Broken 矩阵（按 category）

### 7.1 base → base_math (math RL)
| Category | fixed | broken | net |
|---|---:|---:|---:|
| STEM | 469 | 206 | **+263** |
| Humanities | 766 | 591 | +175 |
| Social Sciences | 616 | 228 | **+388** |
| Other | 570 | 196 | +374 |
| **总计** | **2421** | **1221** | **+1200** |

### 7.2 base_math → final_search (search 训练)
| Category | fixed | broken | net |
|---|---:|---:|---:|
| STEM | 259 | 136 | +123 |
| Humanities | 699 | 458 | **+241** |
| Social Sciences | 248 | 178 | +70 |
| Other | 253 | 137 | +116 |
| **总计** | **1459** | **909** | **+550** |

### 7.3 final_search → tau2（最新阶段）
| Category | fixed | broken | net |
|---|---:|---:|---:|
| STEM | 209 | 103 | **+106** |
| **Humanities** | **580** | **339** | **+241** |
| Social Sciences | 198 | 117 | +81 |
| Other | 168 | 125 | +43 |
| **总计** | **1155** | **684** | **+471** |

→ tau2 阶段 **Humanities 净收益最高**（+241，跟 search 阶段持平）。tau2 在人文类上没有止步。

### 7.4 base → tau2（端到端）
| Category | fixed | broken | net |
|---|---:|---:|---:|
| STEM | 584 | 92 | **+492** |
| Humanities | 1005 | 348 | **+657** |
| Social Sciences | 653 | 114 | +539 |
| Other | 653 | 120 | +533 |
| **总计** | **2895** | **674** | **+2221** |

→ 端到端 fixed:broken = **4.3:1**，每修 4 道才"误伤" 1 道，是所有 transition 中最干净的训练信号。

### 各阶段重点 subject movers（final_search → tau2）

**最大增益**：
- moral_scenarios: 185 fixed / 85 broken = **net +100**
- professional_law: 225 / 156 = +69
- high_school_mathematics: 35 / 6 = +29
- high_school_macroeconomics: 39 / 11 = +28
- formal_logic: 17 / 2 = +15

**最大损失**：
- logical_fallacies: 10 / 16 = -6
- global_facts: 7 / 12 = -5
- computer_security: 5 / 9 = -4
- abstract_algebra: 9 / 10 = -1
- high_school_statistics: 8 / 9 = -1

→ tau2 的损失都是 < -10 的小波动，没有大型回退。

---

## 8. 单题 Pattern 统计（4 ckpt）

| Pattern | 总数 | STEM | Humanities | Social | Other |
|---|---:|---:|---:|---:|---:|
| `stable` | **8678** (61.8%) | 2181 | 2479 | 1964 | 2054 |
| `fixed_at_bm` (✗→✓→✓→✓) | 1850 | 389 | 480 | 501 | 480 |
| `fixed_at_fs` (✗→✗→✓→✓) | 451 | 86 | 224 | 63 | 78 |
| **`fixed_at_tau2`** (✗→✗→✗→✓) | **358** | 68 | **198** | 40 | 52 |
| `broken_at_bm` (✓→✗→✗→✗) | 220 | 21 | 115 | 39 | 45 |
| `broken_at_fs` (✓→✓→✗→✗) | 110 | 12 | 61 | 22 | 15 |
| **`broken_at_tau2`** (✓→✓→✓→✗) | **235** | 43 | **108** | 39 | 45 |
| `volatile_1011` (✓→✗→✓→✓) | **709** | 135 | **310** | 145 | 119 |
| `volatile_1101` (✓→✓→✗→✓) | 378 | 66 | 177 | 79 | 56 |
| `volatile_0101` (✗→✓→✗→✓) | 236 | 41 | 103 | 49 | 43 |
| `volatile_0010` (✗→✗→✓→✗) | 190 | 22 | 101 | 26 | 41 |
| `volatile_0100` (✗→✓→✗→✗) | 185 | 17 | 117 | 28 | 23 |
| `volatile_1001` (✓→✗→✗→✓) | 183 | 34 | 102 | 30 | 17 |
| `volatile_0110` (✗→✓→✓→✗) | 150 | 22 | 66 | 38 | 24 |
| `volatile_1010` (✓→✗→✓→✗) | 109 | 16 | 64 | 14 | 15 |

**关键观察**：
- **`stable` 提高到 61.8%**（3-ckpt 时是 66%；加 tau2 让 8% 题变非稳定，自然 stable 下降不到 5%——挺稳的）
- **`volatile_1011` 数量惊人 (709)**：base 对 → bm 错 → fs 对 → tau2 对 ——这是 base_math 阶段独有的"断层式破坏"模式，tau2 持续修复后保留
- **`fixed_at_tau2` 358 道**：tau2 独立贡献的新 fixed
- **`broken_at_tau2` 235 道**：tau2 独立 broken
- 净 tau2 贡献 = 358 - 235 = **+123 道**（与 final_search → tau2 的 net=+471 不一致是因为 net 算的是相邻两 ckpt，pattern 算的是全 4 ckpt 轨迹）
- **Humanities 在每个 volatile pattern 都占大头**（与 3-ckpt 分析一致）：人文类是最不稳定的类别

---

## 9. Persistent 状态

```
total: 14042
persistent ✓ (4 ckpt 全对): 7463  (53.2%)
persistent ✗ (4 ckpt 全错): 1215  (8.7%)   ← 比 3-ckpt 的 1573 少 358 道！
flapping (变动):           5364   (38.2%)
```

**persistent_wrong 减少 358 道** —— tau2 拿下了之前三 ckpt 都做不对的接近 23% 硬题（358/1573）。

---

## 10. 与之前 3-ckpt 报告的核心结论变化

| 维度 | 3-ckpt 结论 | 加上 tau2 后 |
|---|---|---|
| `boxed` 用法 | math RL 让 boxed 飙到 7.2% (1005 道) | **tau2 又压回 0.3% (39 道)** —— 模型主动卸下 math 训练带来的格式漂移 |
| `no_match` 演化 | base_math 飙升到 5.5%，final_search 回到 3.0% | tau2 进一步降到 **0.7%** —— 几乎清零 |
| Humanities | 涨幅最小 (+8.84pp) | tau2 阶段 +5.12pp，**累计 +13.96pp** —— 人文类不是没在学，是学得慢 |
| moral_scenarios | math RL 阶段 net -48 (退化) | tau2 阶段 net +100，**端到端 net +206** —— 完全反转 |
| `gave_up_short` | base_math 阶段异常高 (502)，final_search 部分恢复 (147) | tau2 进一步降到 **56**，比 base 的 240 还少 |
| 格式不稳定问题 | 中间过渡期 (base_math) 输出乱 | tau2 完全治愈，**回归 base 风格但保留高准确率** |

---

## 11. 跨数据集横向对比（4 ckpt）

| 指标 | math500 | **MMLU** | MMLU-Pro | GPQA |
|---|---:|---:|---:|---:|
| 总题数 | 500 | **14042** | 12032 | 198 |
| base 起点 | 0.7960 | **0.6699** | 0.5677 | 0.3788 |
| base→final_search Δ | +5.6pp | +12.5pp | +13.0pp | +11.6pp |
| final_search→tau2 Δ | +0.6pp | **+3.4pp** ⭐ | (跑中) | -2.0pp |
| **base→tau2 Δ** | +6.4pp | **+15.8pp** ⭐ | (跑中) | +9.6pp |
| persistent_wrong | 41 (8.2%) | 1215 (8.7%) | — | 42 (21.2%) |
| flapping % | 17.4% | 38.2% | — | 64.1% |

**MMLU 是 tau2 vs final_search 涨幅最大的数据集**（+3.4pp），也是 tau2 阶段最干净的训练信号。

---

## 12. 核心结论

1. **tau2 是 MMLU 上最佳 ckpt**（0.8281），比 final_search 净 +3.4pp。这是 4 个数据集中 tau2 阶段涨幅最大的一个。
2. **tau2 同时实现了"分数高 + 格式自然"**：
   - 准确率 0.8281（最高）
   - 但 `\boxed{X}` 用量从 final_search 1005 道压回 39 道（与 base 几乎一样）
   - `the answer is X` 用量从 final_search 77 道恢复到 1157 道（与 base 完全一致）
   - 模型主动卸下 math RL 带来的格式漂移，但保留了准确率收益
3. **格式相关失败几乎全部清零**：
   - `no_match` 104（最低，比 3-ckpt 时最低值还少 75%）
   - `gave_up_short` 56 / `medium_unconverged` 29（跟 final_search 相比都减半以上）
4. **每个 category 都创新高**：
   - STEM 突破 0.9 大关 (0.9052)
   - Humanities 涨到 0.7273（比之前最高 0.6761 高 +5.1pp）
   - Social Sciences 端到端 +17.5pp（最大涨幅）
   - Other 端到端 +17.2pp
5. **moral_scenarios 完全反转**：MMLU 最大且最难的子类，base_math 阶段净 -48，final_search net +154，**tau2 阶段又 net +100**——累计端到端 +23pp
6. **persistent_wrong 减少 358 道**：之前 3 ckpt 都没解决的硬题，tau2 拿下其中 23%
7. **Pattern 统计揭示训练特征**：
   - 358 道 `fixed_at_tau2`（tau2 才修出来的）
   - 235 道 `broken_at_tau2`（tau2 才打破的）
   - 净增 +123 道是 4 ckpt 全程视角看的"tau2 贡献"
   - `volatile_1011` 709 道（base→bm 阶段被 RL 短暂破坏的，到 tau2 都修回来了）
8. **Humanities 仍是最不稳定的 category**：每个 volatile pattern 都占大头，但端到端 +13.96pp 也是不小的涨幅
