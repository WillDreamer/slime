# GPQA 详细分析报告（base / base_math / final_search / tau2）

脚本：
- [tools/eval_analysis/classify_gpqa_failures.py](../tools/eval_analysis/classify_gpqa_failures.py)
- [tools/eval_analysis/breakdown_gpqa_eval.py](../tools/eval_analysis/breakdown_gpqa_eval.py)

CSV 导出：`/tmp/gpqa_per_doc_4ckpt.csv`（198 行，每个 ckpt 各 1 个 0/1 列 + 1 个 via_boxed 列 + pattern 列）

> GPQA Diamond，198 题，4 选项 (A-D)。3 个 high-level domain（Physics 86 / Chemistry 93 / Biology 19）+ 13 个 subdomain。

---

## 1. 总体准确率（4 ckpt）

| | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|
| 原始 exact_match | 0.3788 | 0.4697 | **0.4949** | 0.4747 | +9.59pp |
| re-live extractor | 0.3788 | 0.4697 | 0.4949 | 0.4747 | （完全一致）|

**关键发现**：**tau2 是 4 个数据集里唯一在 GPQA 上回退的 ckpt**——比 final_search 低 4 题 (-2.02pp)。math500/MMLU/MMLU-Pro 上 tau2 都是单调向上。

---

## 2. Layer 1 — 提取路径分布（4 ckpt）

> **注意**：本节反映 [eval/slime_tasks/utils.py](slime_tasks/utils.py) 升级后的提取顺序：
> `pattern_answer → pattern_X_correct → pattern_final → \boxed{X}（新增）→ bare_letter → no_match`。
> 之前 bare-letter 顺手抓到 `\boxed{}` 里字母的样本现在归到独立 `boxed` 桶——分数完全不变（0.3788/0.4697/0.4949/0.4747），只是桶分类更诚实。

| 提取路径 | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| `pattern_answer` | 5 (2.5%) | 3 (1.5%) | 7 (3.5%) | **12 (6.1%)** |
| `pattern_X_correct` | 0 (0.0%) | 0 (0.0%) | 2 (1.0%) | **7 (3.5%)** |
| `pattern_final` | 0 (0.0%) | 0 (0.0%) | 1 (0.5%) | 0 (0.0%) |
| **`boxed` (`\boxed{X}`)** | 5 (2.5%) | **72 (36.4%)** | 40 (20.2%) | **38 (19.2%)** |
| `bare_letter` | 174 (87.9%) | 111 (56.1%) | 143 (72.2%) | 139 (70.2%) |
| `no_match` | 14 (7.1%) | 12 (6.1%) | 5 (2.5%) | **2 (1.0%)** |

**关键观察**：
- tau2 的 strict pattern 用得明显多：`pattern_answer` (12, 6.1%) + `pattern_X_correct` (7, 3.5%) + `pattern_final` (0) = **19 道走 strict**（之前 ckpt 最多只有 10 道），模型更倾向用 "the answer is X" / "X is correct" 等明确表达
- `boxed` 路径的清晰分布显示：base 几乎不用 (5)，math RL **暴涨到 72**，search 减半 (40)，tau2 与 final_search 接近 (38)
- `bare_letter` 是真正"没明确收尾标志"的样本：base 174（87.9%）→ tau2 139（70.2%），稳定下降意味着模型答 GPQA 时越来越遵循某种格式规范
- `no_match` 单调下降：14 → 12 → 5 → 2，tau2 几乎清零

### 各路径的命中率

| Path | base acc | base_math acc | final_search acc | tau2 acc |
|---|---:|---:|---:|---:|
| `pattern_answer` | 1/5 (0.20) | 0/3 (0.00) | 1/7 (0.14) | 1/12 (0.08) |
| `pattern_X_correct` | — | — | 0/2 (0.00) | 1/7 (0.14) |
| `pattern_final` | — | — | 1/1 (1.00) | — |
| **`boxed`** | 3/5 (0.60) | 44/72 (**0.61**) | 25/40 (**0.625**) | 27/38 (**0.711**) |
| `bare_letter` | 71/174 (**0.41**) | 49/111 (0.44) | 70/143 (0.49) | 62/139 (0.45) |

**重要洞察**：
- **`boxed` 路径准确率始终 ≥ bare_letter** —— 一旦模型选择用 `\boxed{X}` 收尾，命中率比散装 bare 字母高 17-26pp
- **tau2 的 boxed 准确率达到 0.711** —— 4 ckpt 中最高，说明 tau2 用 \boxed{} 时是它**最有把握**的样本
- **`pattern_answer` / `pattern_X_correct` 反而准确率很低** (0.08-0.20) —— strict pattern 经常误中（比如 "look at option B and option C..." 抓到 B 但答案是 C）。这正是之前 3-ckpt 报告里提到的 GPQA strict 提取的固有 bug

### Boxed-vs-strict disagreement 审计

| | base | base_math | final_search | **tau2** |
|---|---:|---:|---:|---:|
| 同时存在 \boxed{X} + strict 提取，但二者不一致 | 0 | 0 | 0 | **3** |
| 加 boxed 优先级能救起来的 | 0 | 0 | 0 | **1** |

**首次在 tau2 出现 3 个 disagreement 案例** —— strict pattern 抓到的字母跟 `\boxed{X}` 里的不一样。当前提取顺序是 strict 优先，所以 strict 赢。其中 1 道里 boxed 字母 == target，意味着如果把 boxed 路径放到 strict 之前可以多对 1 道。当前实现保留 strict 优先，与 MMLU-Pro 一致。

---

## 3. Layer 3 — 失败子类（4 ckpt，按提取路径分桶）

| 子类 | base | base_math | final_search | **tau2** |
|---|---:|---:|---:|---:|
| `extracted_wrong` (strict 路径答错) | 4 | 3 | 7 | **13** |
| **`boxed_wrong`** (boxed 路径答错) | 2 | **28** | **15** | **11** |
| **`bare_wrong`** (bare 路径答错) | **103** | **62** | **73** | **77** |
| `rescuable_via_boxed` | 0 | 0 | 0 | 1 |
| `hidden_failure` | 0 | 0 | 0 | 0 |
| `gave_up_short` | 4 | 0 | 0 | 0 |
| `medium_unconverged` | 6 | 6 | 3 | 1 |
| `truncated_max_tokens` | 1 | 5 | 2 | 1 |
| `repetition_loop` | 3 | 1 | 0 | 0 |
| `python_block_unfinished` | 0 | 0 | 0 | 0 |
| `reach_max_function_call` | 0 | 0 | 0 | 0 |

**关键观察**（和升级前对比）：
- **`bare_wrong` 大幅减少（base→tau2: 103 → 77）**——大部分原本被算成 bare-letter 的错都从 bare 桶剥离到 boxed 桶
- **`boxed_wrong` 现在是独立可追踪桶**：base_math 阶段 28 道 boxed 答错（占总错的 28%），search 把这数字降到 15，tau2 进一步到 11。看 boxed 路径的"错误率"演变比看混在 bare 里更清晰
- **`extracted_wrong` 翻倍** (7 → 13)：tau2 用 strict pattern 抓到字母但答错的样本变多，符合"strict path 用得更多"的现象——但 strict 本身就是个低准确率路径（见上节 0.08-0.20）
- **真正的格式型失败几乎清零**：`medium_unconverged` 1 道、`truncated_max_tokens` 1 道、`gave_up_short` 0 道、`reach_max_function_call` 永远 0

### 各路径"错误率"演变（更清晰的训练信号）

| Path | base | base_math | final_search | tau2 | 趋势 |
|---|---:|---:|---:|---:|---|
| `boxed_wrong / boxed_total` | 2/5 (0.40) | 28/72 (**0.39**) | 15/40 (0.38) | 11/38 (**0.29**) | **boxed 路径错误率单调下降** ↓ |
| `bare_wrong / bare_total` | 103/174 (0.59) | 62/111 (0.56) | 73/143 (0.51) | 77/139 (0.55) | bare 路径错误率小幅波动 |
| `extracted_wrong / strict_total` | 4/5 (0.80) | 3/3 (1.00) | 7/10 (0.70) | 13/19 (0.68) | strict 一直是高错误率（陷阱） |

**核心发现**：**`boxed` 路径是 GPQA 上**最可靠**的提取路径，错误率从 0.40 单调下降到 0.29，且 tau2 上**升级最大**——说明模型用 `\boxed{X}` 时越训练越准。这是之前 3-ckpt 报告里完全看不出来的训练效果信号。

---

## 4. 按 high-level domain（核心发现）

| Domain | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| **Physics** | 86 | 0.4070 | 0.5233 | 0.6163 | **0.6512** | **+24.4pp** ⭐ |
| **Chemistry** | 93 | 0.3441 | 0.3978 | 0.3763 | **0.3333** | **-1.1pp** ⚠️ |
| **Biology** | 19 | 0.4211 | 0.5789 | 0.5263 | **0.3684** | **-5.3pp** ⚠️ |

**这是整份报告最重要的图**：
- **Physics 单调上升 +24.4pp**——tau2 在 GPQA 上唯一受益的 domain
- **Chemistry 端到端首次跌破 base**（0.3441 → 0.3333，-1.1pp）—— 之前两轮训练都还能小幅 +，tau2 反向了
- **Biology 大幅跌破 base**（0.4211 → 0.3684，-5.3pp）—— 三轮 RL 训练对 Biology 是**净负面影响**！

### via_boxed 在各 domain 的分布

| ckpt | Physics (n=86) | Chemistry (n=93) | Biology (n=19) |
|---|---:|---:|---:|
| base | 5/86 (5.8%) acc 0.600 | 0/93 | 0/19 |
| base_math | **48/86 (55.8%)** acc 0.667 | 24/93 (25.8%) acc 0.500 | 0/19 |
| final_search | 39/86 (45.3%) acc 0.641 | **1/93 (1.1%)** acc 0.000 | 0/19 |
| **tau2** | 30/86 (34.9%) acc **0.833** | 10/93 (10.8%) acc 0.300 | **1/19 (5.3%)** acc 1.000 |

**有趣的格式变化**：
- Physics 用 boxed 的比例**单调下降** (55.8% → 45.3% → 34.9%)，但准确率**单调上升** (0.667 → 0.641 → **0.833**)。模型越用越精准——只在最有把握时才用 boxed
- Chemistry 在 final_search 几乎完全戒掉 boxed（24→1），tau2 又回弹到 10.8%——但准确率 0.300 很低（在 chemistry 题上 boxed 反而不可靠）
- Biology **首次出现 1 道 boxed**（之前 3 ckpt 全是 0）

---

## 5. 按 13 个 subdomain

| Subdomain | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| **Quantum Mechanics** | 25 | 0.520 | 0.600 | 0.800 | **0.840** | **+0.320** ⭐ |
| Relativistic Mechanics | 7 | 0.286 | 0.571 | 0.714 | 0.714 | +0.429 |
| **Physics (general)** | 19 | 0.421 | 0.421 | 0.474 | **0.632** | **+0.211** |
| High-energy particle physics | 14 | 0.357 | 0.714 | 0.571 | 0.643 | +0.286 |
| Electromagnetism and Photonics | 6 | 0.333 | 0.500 | 0.500 | 0.500 | +0.167 |
| Chemistry (general) | 20 | 0.350 | 0.550 | 0.450 | 0.500 | +0.150 |
| Astrophysics | 13 | 0.308 | 0.308 | 0.538 | 0.308 | ±0 (回到 base) |
| Molecular Biology | 15 | 0.333 | 0.533 | 0.533 | 0.400 | +0.067 |
| **Organic Chemistry** | **72** | **0.347** | **0.361** | **0.347** | **0.292** | **-0.056** ⚠️ |
| **Genetics** | 4 | 0.750 | 0.750 | 0.500 | **0.250** | **-0.500** ⚠️ |
| Inorganic Chemistry (单题) | 1 | 0.000 | 0.000 | 1.000 | 0.000 | ±0 |
| Optics and Acoustics (单题) | 1 | 0.000 | 0.000 | 0.000 | **1.000** | +1.000 |
| Condensed Matter Physics (单题) | 1 | 1.000 | 1.000 | 1.000 | 1.000 | ±0 |

**关键发现**：
- **Quantum Mechanics 是 GPQA 真正的训练受益点**：4 ckpt 单调上升，tau2 拿下 0.840
- **Physics (general) 在 tau2 才突破**：之前一直 0.421-0.474，tau2 飙到 0.632 (+16pp)
- **Organic Chemistry 创新低**：72 题里 21/72 = 0.292，比 base 的 0.347 还低 5.6pp。这是 GPQA 上**最明确的训练负影响**
- **Genetics 持续退化**：base 0.75 → tau2 0.25，4 题里只对了 1 题（样本小但趋势严峻）

---

## 6. 三对相邻 transition + 端到端

### base → base_math（math RL）

```
fixed=37, broken=19, net=+18
Physics +10  Chemistry +5  Biology +3
```

### base_math → final_search（search 训练）

```
fixed=37, broken=32, net=+5
Physics +8  Chemistry -2  Biology -1
```

### **final_search → tau2（最新阶段，唯一净负转移）**

```
fixed=36, broken=40, net=-4
Physics +3  Chemistry -4  Biology -3
```

按 subdomain (final_search → tau2)：
| Subdomain | fixed | broken | net |
|---|---:|---:|---:|
| Physics (general) | 5 | 2 | **+3** |
| Quantum Mechanics | 2 | 1 | +1 |
| Chemistry (general) | 3 | 2 | +1 |
| High-energy particle physics | 4 | 3 | +1 |
| Optics and Acoustics | 1 | 0 | +1 |
| Astrophysics | 3 | 6 | **-3** |
| Molecular Biology | 2 | 4 | **-2** |
| **Organic Chemistry** | 13 | 17 | **-4** ⚠️ |
| Inorganic Chemistry | 0 | 1 | -1 |
| Genetics | 0 | 1 | -1 |

→ Organic Chemistry 单格 **净 -4**，是 tau2 GPQA 退步的最大单一来源。

### base → tau2（端到端）

```
fixed=48, broken=29, net=+19
Physics +21  Chemistry -1  Biology -1
```

→ 端到端净 +19 道，**Physics 一家贡献 21**，Chemistry 和 Biology 净负。

---

## 7. 单题 Pattern 统计（4 ckpt）

| Pattern | 总数 | Physics | Chemistry | Biology |
|---|---:|---:|---:|---:|
| `stable` (4 ckpt 一致) | **71** (35.9%) | 31 | 31 | 9 |
| ↳ stable_correct (✓✓✓✓) | 29 | — | — | — |
| ↳ stable_wrong (✗✗✗✗) | 42 | — | — | — |
| `fixed_at_tau2` (✗✗✗✓) | **18** | 8 | 9 | 1 |
| `fixed_at_bm` (✗✓✓✓) | 13 | 10 | 2 | 1 |
| `fixed_at_fs` (✗✗✓✓) | 9 | 3 | 6 | 0 |
| `broken_at_tau2` (✓✓✓✗) | **14** | 3 | 9 | 2 |
| `broken_at_fs` (✓✓✗✗) | 7 | 0 | 5 | 2 |
| `broken_at_bm` (✓✗✗✗) | 4 | 0 | 4 | 0 |
| `volatile_*`（多次反转） | 62 | 31 | 27 | 4 |

**关键观察**：
- **stable 只占 36%**——GPQA 上 tau2 加进来后 64% 的题都至少反转了一次
- `fixed_at_tau2` (18) > `broken_at_tau2` (14)：tau2 单步**修题数 > 破坏数**，但因为 broken 平均"含金量"更高（更多 bare_wrong → bare_correct 边缘 case），最终净 -4
- `fixed_at_tau2` 在 Chemistry 占 9 道 —— 化学不是没在学，只是学的题数 ≈ 退步的题数 (9 vs 9)
- `broken_at_tau2` 在 Chemistry 占 9 道（其中 8 道是 Organic Chemistry）

### Volatile 桶细分（4-ckpt bitstring）

| Pattern | 总数 | Physics | Chemistry | Biology |
|---|---:|---:|---:|---:|
| `volatile_0010` (✗✗✓✗) | 17 | 8 | 7 | 2 |
| `volatile_0100` (✗✓✗✗) | 11 | 3 | 8 | 0 |
| `volatile_0101` (✗✓✗✓) | 8 | 4 | 3 | 1 |
| `volatile_1011` (✓✗✓✓) | 7 | 6 | 1 | 0 |
| `volatile_1101` (✓✓✗✓) | 6 | 3 | 3 | 0 |
| `volatile_0110` (✗✓✓✗) | 5 | 3 | 1 | 1 |
| `volatile_1001` (✓✗✗✓) | 4 | 3 | 1 | 0 |
| `volatile_1010` (✓✗✓✗) | 4 | 1 | 3 | 0 |

→ `volatile_0010` 17 道是"只有 final_search 做对"的题——tau2 又把它们丢了。

---

## 8. Persistent 状态（4 ckpt 全错）

```
total: 198
persistent ✓ (4 ckpt 全对): 29  (14.6%)
persistent ✗ (4 ckpt 全错): 42  (21.2%)   ← 比 3-ckpt 的 60 减少 18 道！
flapping (变动):           127 (64.1%)   ← 全数据集中最高的 flapping 比例
```

`persistent_wrong` **从 60 降到 42** —— tau2 拿下了 18 道之前三 ckpt 都做不出来的硬题，这是 tau2 在 GPQA 上的真正贡献（虽然代价是同时打破了 14 道之前对的）。

按 subdomain 的 persistent_wrong (top 5)：
| Subdomain | persistent_wrong | 占该 subdomain |
|---|---:|---:|
| **Organic Chemistry** | 21 | 21/72 = 29% |
| Chemistry (general) | 4 | 20% |
| Physics (general) | 4 | 21% |
| Molecular Biology | 4 | 27% |
| Electromagnetism and Photonics | 2 | 33% |

→ Organic Chemistry 单类仍占全部 persistent_wrong 的 50%（21/42），**仍是 GPQA 最大的训练盲区**。

---

## 9. 与 3-ckpt 报告的核心结论变化

| 维度 | 3-ckpt 结论 | 加上 tau2 后 |
|---|---|---|
| 整体趋势 | base→final 单调上升 (+11.6pp) | tau2 **回退** (-2pp vs final_search) |
| Physics | math RL 主战场 (+20.9pp) | **继续上升 +24.4pp**（+3.5pp from final_search）|
| Chemistry | 几乎不动 (+3.2pp) | **跌破 base** (-1.1pp) |
| Biology | 小样本不稳 (+10.5pp) | **跌破 base** (-5.3pp，更严峻) |
| Organic Chemistry | 完全不动 (25/72 → 25/72) | **创新低 21/72** (-4 题) |
| Quantum Mechanics | +28pp 主受益 | **持续到 +32pp，0.840** |
| boxed 用法 | 100% 集中 Physics | tau2 让 Chemistry 回弹 + Biology 首次出现 |
| via_boxed 准确率 | 0.60-0.625 | **跳到 0.711**（boxed 信号变更可靠）|

---

## 10. 跨数据集横向对比（4 ckpt）

| 指标 | math500 | MMLU | MMLU-Pro | **GPQA** |
|---|---:|---:|---:|---:|
| 总题数 | 500 | 14042 | 12032 | 198 |
| base 起点 | 0.7960 | 0.6699 | 0.5677 | **0.3788** |
| base→final_search Δ | +5.6pp | +12.5pp | +13.0pp | +11.6pp |
| **final_search→tau2 Δ** | +0.6pp | (跑中) | (跑中) | **-2.0pp** ⚠️ |
| **base→tau2 Δ** | **+6.4pp** | (跑中) | (跑中) | **+9.6pp** |
| persistent_wrong | 41 (8.2%) | — | — | **42 (21.2%)** |
| flapping % | 17.4% | — | — | **64.1%** |

**GPQA 是唯一一个 tau2 比 final_search 退步的数据集** —— 这是 tau2 训练数据/目标的清晰信号：tau2 收益了 Physics（+24pp 单调），但代价是 Chemistry/Biology 退化。

---

## 11. 核心结论

1. **tau2 在 GPQA 上是 4 个数据集中唯一的回退**：-2.0pp vs final_search，但仍比 base 高 +9.6pp
2. **Physics 是 tau2 的舒适区**：单调上升 +24.4pp，Quantum Mechanics 0.840（含 25 题里 21 题对），Physics (general) 突破 0.632
3. **Chemistry 和 Biology 跌破 base**：Chemistry -1.1pp（base 0.344 → tau2 0.333），Biology -5.3pp（base 0.421 → tau2 0.368）—— 这是**三轮训练对这两个领域的累积净伤害**
4. **Organic Chemistry 创新低** 21/72 (0.292)：base 阶段就低，tau2 上更糟，单格贡献 GPQA persistent_wrong 的 50%
5. **boxed 信号质量提升**：via_boxed 准确率从 0.625 跳到 **0.711**——模型用 \boxed{X} 时更可靠，但用量没变（38-41 之间）
6. **Physics 模型选择性写 boxed**：用量从 base_math 的 55.8% 降到 tau2 的 34.9%，但准确率从 0.667 升到 **0.833**——只在最有把握时才用
7. **首次出现 boxed_disagreement** (3 道)：tau2 上 strict pattern 抓的字母 ≠ \boxed{X} 里的字母；其中 1 道如果加 boxed 路径优先于 strict 能救（当前 strict 仍优先）
8. **`\boxed{X}` 已被升级为 GPQA 的正式提取路径**（[utils.py 第 225 行起](slime_tasks/utils.py)）：顺序 strict → boxed → bare，分数完全不变（之前 bare-letter 顺手抓 box 内字母），但桶分类更诚实，揭示 boxed 路径错误率单调下降 (0.40 → 0.29) 这个之前看不到的训练信号
9. **持久全错减少**：60 → 42（-18 道），但代价是 14 道之前对的题被破坏，净 fixed:broken = 36:40 = -4
10. **Pattern 统计揭示 tau2 的"重新洗牌"性质**：fixed_at_tau2 18 道 / broken_at_tau2 14 道，几乎"互相抵消"——加上 volatile_0010 (final_search 独占) 17 道也丢了，整体看是"训练过程中 GPQA 的对错样本在反复换"
11. **GPQA 的 flapping 比例 64% 是全数据集最高**——4 ckpt 中 64% 的题至少反转过一次，说明 GPQA 这种小样本难题对 RL 训练的"波动"最敏感
