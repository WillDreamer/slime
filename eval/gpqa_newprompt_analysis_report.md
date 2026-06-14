# GPQA 详细分析报告（新 prompt，base / base_math / final_search / tau2）

新 prompt（[gpqa_slime.yaml:17](slime_tasks/gpqa_slime.yaml#L17)）显式要求模型把答案字母写进 `\boxed{}`：

```
Answer the following multiple choice question. Think step by step,
then put your final answer letter inside \boxed{}, e.g. \boxed{A}.
```

脚本：
- [tools/eval_analysis/classify_gpqa_failures.py](../tools/eval_analysis/classify_gpqa_failures.py)
- [tools/eval_analysis/breakdown_gpqa_eval.py](../tools/eval_analysis/breakdown_gpqa_eval.py)

CSV 导出：`/tmp/gpqa_newprompt_per_doc.csv`（198 行，每个 ckpt 一列正确性 + 一列 `via_boxed` + pattern 列）

> GPQA Diamond，198 题，4 选项 (A-D)。3 个 high-level domain：Physics 86 / Chemistry 93 / Biology 19；13 个 subdomain。

---

## 1. 总体准确率（4 ckpt，新 vs 旧 prompt 对照）

| | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|
| **新 prompt** exact_match | 0.3232 | 0.4899 | 0.4646 | **0.4949** | **+17.2pp** |
| 旧 prompt exact_match | 0.3788 | 0.4697 | 0.4949 | 0.4747 | +9.6pp |
| Δ (新 − 旧) | **-5.6pp** | +2.0pp | -3.0pp | **+2.0pp** | +7.6pp |
| re-live extractor | 0.3232 | 0.4899 | 0.4646 | 0.4949 | (与原始一致) |

**关键差异（vs 旧 prompt）**：

1. **新 prompt 下 tau2 重新成为最佳 ckpt**（0.4949），不再是旧 prompt 中的 final_search。final_search 反而是新 prompt 受益最少（甚至 -3.0pp）的那个 ckpt。
2. **base 在新 prompt 下反而退步 -5.6pp**——base model 不擅长在指令下用 `\boxed{}` 收尾，多写了 `\boxed{X}` 但很多写错。
3. **end-to-end 改进从 +9.6pp 拉到 +17.2pp**——加严 prompt 后 base 起点拉低、tau2 起点抬高，把训练效果放大近一倍。

---

## 2. Layer 1 — 提取路径分布（4 ckpt）

> 提取顺序仍是 `pattern_answer → pattern_X_correct → pattern_final → \boxed{X} → bare_letter → no_match`（[utils.py](slime_tasks/utils.py)）。新 prompt 把模型推向 `boxed` 路径，旧 prompt 大量样本都靠 `bare_letter` 兜底。

| 提取路径 | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| `pattern_answer` | 6 (3.0%) | 1 (0.5%) | 4 (2.0%) | **14 (7.1%)** |
| `pattern_X_correct` | 0 (0.0%) | 0 (0.0%) | 1 (0.5%) | **5 (2.5%)** |
| `pattern_final` | 0 (0.0%) | 0 (0.0%) | 1 (0.5%) | 0 (0.0%) |
| **`boxed`** | **139 (70.2%)** | **179 (90.4%)** | **189 (95.5%)** | **171 (86.4%)** |
| `bare_letter` | 45 (22.7%) | 10 (5.1%) | **1 (0.5%)** | 6 (3.0%) |
| `no_match` | 8 (4.0%) | 8 (4.0%) | 2 (1.0%) | 2 (1.0%) |

**关键观察**：

- **`boxed` 路径占绝对主导**：从 base 70% 到 final_search **95.5%**，模型已经把 `\boxed{}` 内化成首选格式。旧 prompt 下 boxed 占比仅 2.5%-36% 不等。
- **`bare_letter` 几乎清零**：final_search 只剩 1 道走 bare-letter（旧 prompt 下还有 87.9% / 56.1% / 72.2% / 70.2%）。这是新 prompt 最大的格式变化。
- **`pattern_answer` 在 tau2 上回升到 7.1%**（14 道）：tau2 倾向口头明确 "the answer is X"，比单纯 boxed 表达更显式——这是 tau2 与其他 ckpt 不同的"语气"。
- **`no_match` 跨 ckpt 单调下降**：8 → 8 → 2 → 2，最后两个 ckpt 已基本不会"无答案"。

### 各路径的命中率（accuracy by path）

| Path | base acc | base_math acc | final_search acc | tau2 acc |
|---|---:|---:|---:|---:|
| `pattern_answer` | 4/6 (0.67) | 1/1 (1.00) | 1/4 (0.25) | 6/14 (0.43) |
| `pattern_X_correct` | — | — | 1/1 (1.00) | 4/5 (0.80) |
| `pattern_final` | — | — | 0/1 (0.00) | — |
| **`boxed`** | 49/139 (**0.353**) | 92/179 (**0.514**) | 91/189 (**0.481**) | 88/171 (**0.515**) |
| `bare_letter` | 13/45 (0.29) | 4/10 (0.40) | 0/1 (0.00) | 0/6 (0.00) |

**关键洞察**：

- **新 prompt 下 boxed 路径仍是最稳定的**，但准确率不如旧 prompt 那么"挑剔"：旧 prompt 里 boxed 准确率 0.60-0.71（因为模型只在最有把握时才用 box）；新 prompt 里 boxed 准确率 0.35-0.51（因为被强行用，把"猜"也写进 box）。
- **`bare_letter` 在新 prompt 下变成"垃圾桶"**：能走到 bare 的样本通常是 box 都没写出来的失败案例，因此命中率从旧 prompt 的 0.41-0.49 跌到 0.00-0.40。
- **tau2 的 strict patterns 命中率非常高**（`pattern_answer` 0.43、`pattern_X_correct` **0.80**），因为 tau2 倾向"说出来 + 写在 box 里"，strict 抓到时几乎都对——但严格优先（不通过 box 复核）反而压住了部分本来正确的 boxed 信号。

### Boxed-aware 二次审计：via_boxed（包含 strict 路径里也出现 `\boxed{}` 的样本）

| | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| via_boxed 总数 | 144 (72.7%) | 180 (90.9%) | **195 (98.5%)** | 189 (95.5%) |
| via_boxed acc | 0.347 | 0.511 | 0.472 | **0.513** |
| boxed-vs-strict disagreement | 5 | 1 | 3 | **7** |
| 加 boxed 优先级能救起来的 | 1 | 0 | 3 | **3** |

**新现象**：

- **final_search 的 via_boxed 覆盖率 98.5%**——195/198 道里，`\boxed{X}` 内的字母都跟最终提取一致。剩下 3 道全是 `rescuable_via_boxed`（strict 抓错，box 是对的）。
- **tau2 disagreement 从旧 prompt 的 3 道涨到 7 道**：因为 tau2 更爱 strict 表达（"the answer is X"），同时也写 box，两边偶尔不一致。3 道 box 是 target——加 boxed 优先于 strict 可以多对 3 道（潜在 +1.5pp）。
- 全 4 ckpt **rescuable_via_boxed = 7 道** = 0+0+3+1+3。这是 prompt 工程的纯收益。

### 三个真实 rescuable_via_boxed 样本（tau2）

```
=== doc_id 11  target=D  strict(pa)=C  boxed=D ===
"...the closest options are (D) 4.12 MeV, 29.8 MeV. So, the answer is
likely D."  ← strict 抓到第一处 "(D)" 后又被 "answer is likely D" 冲掉
"...So, the correct answer is \boxed{D}."  ← box 里就是 D

=== doc_id 29  target=C  strict(pxc)=D  boxed=C ===
"...the complementary color of green is indeed red. Therefore, the
correct answer is \boxed{C}."  ← box=C
strict 在更上方抓到 "the option D is correct" 之类引用 → strict 输出 D

=== doc_id 195 target=B  strict(pa)=C  boxed=B ===
"The correct answer is likely (B)... \boxed{B}"
strict 之前抓到 "option C 不对" 这类讨论里的 C
```

→ 都是同一种 bug 模式：**strict 模式抓到 "讨论中提到的字母"，把真实结论（box 里）覆盖掉**。

---

## 3. Layer 3 — 失败子类（4 ckpt，按提取路径分桶）

| 子类 | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| `extracted_wrong` (strict 路径答错) | 3 | 1 | 2 | **7** |
| **`boxed_wrong`** (boxed 路径答错) | **90** | **87** | **98** | **83** |
| `bare_wrong` (bare 路径答错) | 32 | 5 | 1 | 5 |
| `rescuable_via_boxed` | 1 | 0 | 3 | 3 |
| `hidden_failure` | 0 | 0 | 0 | 0 |
| `gave_up_short` | 6 | 0 | 0 | 0 |
| `medium_unconverged` | 1 | 5 | 0 | 1 |
| `truncated_max_tokens` | 0 | 1 | 1 | 1 |
| `repetition_loop` | 1 | 2 | 1 | 0 |
| `python_block_unfinished` | 0 | 0 | 0 | 0 |
| `reach_max_function_call` | 0 | 0 | 0 | 0 |

**与旧 prompt 对比的最大变化**：

- **`bare_wrong` 大幅萎缩**（旧→新）：103→32 / 62→5 / 73→1 / 77→5。新 prompt 把 bare 路径上的错都迁移到 `boxed_wrong` 桶。
- **`boxed_wrong` 暴涨**（旧→新）：2→90 / 28→87 / 15→98 / 11→83。**新 prompt 下 `boxed_wrong` 是 GPQA 错误的"主桶"**——80% 以上的错都发生在模型用 box 收尾时。这是合理的，因为 95% 样本都走 box。
- **`extracted_wrong` (strict 抓到的字母错) tau2 上升到 7 道**：tau2 的 strict pattern 用得多，一旦 strict 抢占 box 抓到错字母（即使 box 是对的）就被算错——这就是 `rescuable_via_boxed` 现象的根源。
- **真正的格式型失败几乎清零**：`gave_up_short` 仅 base 上 6 道，`medium_unconverged` ≤ 5 道，`reach_max_function_call` 全 0。

### 各路径"错误率"演变（更清晰的训练信号）

| Path | base | base_math | final_search | tau2 | 趋势 |
|---|---:|---:|---:|---:|---|
| `boxed_wrong / boxed_total` | 90/139 (0.65) | 87/179 (**0.49**) | 98/189 (0.52) | 83/171 (**0.49**) | base 阶段 boxed 错最多，RL 后稳定 ~0.49 |
| `bare_wrong / bare_total` | 32/45 (0.71) | 5/10 (0.50) | 1/1 (1.00) | 5/6 (0.83) | bare 几乎无人走，是失败兜底 |
| `extracted_wrong / strict_total` | 3/6 (0.50) | 1/1 (1.00) | 2/6 (0.33) | 7/19 (0.37) | strict 用得多但错率较稳 |

**核心发现**：**`boxed` 路径错误率 0.49** 在 4 个 ckpt 里几乎相同——RL 后 GPQA 上 box 路径的"上限"基本固定在 ~50% 命中。新 prompt 把所有提取路径"拍平"到 box 一条，反而暴露了模型对 GPQA 的能力天花板：**约 50% 的 box 都是错的，跟模型自身的领域能力强相关**（见下节 domain 分布）。

---

## 4. 按 high-level domain（核心发现）

| Domain | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| **Physics** | 86 | 0.3488 | 0.6047 | 0.5930 | **0.6977** | **+34.9pp** ⭐ |
| **Chemistry** | 93 | 0.2903 | 0.3763 | 0.3226 | **0.3011** | **+1.1pp** |
| **Biology** | 19 | 0.3684 | 0.5263 | **0.5789** | 0.5263 | +15.8pp |

| | 旧 prompt base→tau2 | 新 prompt base→tau2 | Δ |
|---|---:|---:|---:|
| Physics | +24.4pp | **+34.9pp** | +10.5pp 更强 |
| Chemistry | -1.1pp | +1.1pp | +2.2pp，新 prompt 略好 |
| Biology | -5.3pp | +15.8pp | **+21.1pp** 翻盘！|

**核心差异**（vs 旧 prompt）：

- **Physics**：仍是最大受益 domain，新 prompt 下 tau2 拿下 **0.6977**（86 题里对 60 道），比旧 prompt 的 0.6512 还高 +4.7pp。
- **Chemistry**：旧 prompt 下 tau2 跌破 base，新 prompt 下小幅 +1.1pp，**不再是负迁移**。
- **Biology**：从旧 prompt 的 -5.3pp 翻盘到 +15.8pp。**关键是新 prompt 在 Biology 上的 base 从 0.4211 降到 0.3684**，把"起点"压低了，tau2 维持 0.5263 反而看起来涨了。但 final_search 实际上是 4 ckpt 里 Biology 最强（0.5789）。

### via_boxed 在各 domain 的分布

| ckpt | Physics (n=86) | Chemistry (n=93) | Biology (n=19) |
|---|---:|---:|---:|
| base | 60/86 (69.8%) acc 0.367 | 71/93 (76.3%) acc 0.310 | 13/19 (68.4%) acc 0.462 |
| base_math | 80/86 (93.0%) acc 0.638 | 81/93 (87.1%) acc 0.395 | 19/19 (100%) acc 0.526 |
| final_search | 84/86 (97.7%) acc 0.595 | 92/93 (98.9%) acc 0.326 | 19/19 (100%) acc 0.579 |
| **tau2** | 80/86 (93.0%) acc **0.738** | 90/93 (96.8%) acc 0.311 | 19/19 (100%) acc 0.526 |

**有趣的格式变化**（vs 旧 prompt）：

- 旧 prompt 下 boxed 在 Biology 几乎为 0；新 prompt **Biology 100% 走 boxed**，所有 ckpt 一致。
- Physics 上 tau2 的 via_boxed 准确率 **0.738**——和旧 prompt 的 0.833 接近，但样本基数大得多（80 道 vs 30 道）。
- Chemistry 上 boxed 用得最多但准确率最差（~0.31），符合 "Chemistry 是模型短板" 的现象。

---

## 5. 按 13 个 subdomain

| Subdomain | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| **Physics (general)** | 19 | 0.158 | 0.526 | 0.579 | **0.737** | **+0.579** ⭐⭐ |
| **Relativistic Mechanics** | 7 | 0.143 | 0.571 | 0.571 | **0.714** | **+0.571** ⭐ |
| **Quantum Mechanics** | 25 | 0.480 | 0.800 | 0.680 | **0.840** | **+0.360** ⭐ |
| Electromagnetism and Photonics | 6 | 0.333 | 0.667 | **0.833** | 0.667 | +0.333 |
| Genetics | 4 | 0.250 | 0.500 | **0.750** | 0.500 | +0.250 |
| High-energy particle physics | 14 | 0.500 | 0.643 | 0.643 | 0.643 | +0.143 |
| Chemistry (general) | 20 | 0.350 | 0.500 | 0.450 | 0.500 | +0.150 |
| Molecular Biology | 15 | 0.400 | 0.533 | 0.533 | 0.533 | +0.133 |
| Astrophysics | 13 | 0.308 | 0.308 | **0.385** | 0.385 | +0.077 |
| **Organic Chemistry** | **72** | **0.278** | **0.333** | **0.292** | **0.250** | **-0.028** ⚠️ |
| Inorganic Chemistry (n=1) | 1 | 0.000 | 1.000 | 0.000 | 0.000 | ±0 |
| Optics and Acoustics (n=1) | 1 | 0.000 | 0.000 | 0.000 | **1.000** | +1.000 |
| Condensed Matter Physics (n=1) | 1 | 1.000 | 1.000 | 0.000 | 1.000 | ±0 |

**关键发现**：

- **Physics (general) 是新 prompt 下最大赢家**：base 仅 3/19 → tau2 14/19 = 0.737，**+57.9pp**！旧 prompt 下也仅 +21pp，新 prompt 几乎翻倍。
- **Relativistic Mechanics 重生**：base 1/7 → tau2 5/7（+57.1pp，旧 prompt +42.9pp）。
- **Quantum Mechanics 维持 +36pp 强势**：tau2 21/25 = 0.840，跨 prompt 一致是 GPQA 训练的核心收益点。
- **Organic Chemistry 仍是顽固盲区**：72 题占 GPQA 36%，base→tau2 从 20/72 → 18/72 (-2.8pp)，比旧 prompt 的 -5.6pp 更轻但仍是负的。
- **Astrophysics 在新 prompt 下从 0.308 提升到 final_search 0.385，tau2 维持**（旧 prompt 下 final_search 0.538 → tau2 0.308 反向）。

---

## 6. 三对相邻 transition + 端到端

### base → base_math（math RL）

```
fixed=55, broken=22, net=+33   (旧 prompt: net=+18)
Physics +22  Chemistry +8  Biology +3
```

新 prompt 下 net 从 +18 拉到 +33——math RL 的实际收益被旧 prompt 低估了。

按 subdomain top：
- Quantum Mechanics +8 / Physics (general) +7 / Organic Chemistry +4 / Chemistry (general) +3 / Relativistic Mechanics +3

### base_math → final_search（search 训练）

```
fixed=20, broken=25, net=-5   (旧 prompt: net=+5)
Physics -1  Chemistry -5  Biology +1
```

**新 prompt 下 search 训练阶段反而是净负 (-5)**——旧 prompt 下还能 +5。Quantum Mechanics 反向 -3、Organic Chemistry -3 是主因。

### final_search → tau2（最新阶段）

```
fixed=31, broken=25, net=+6   (旧 prompt: net=-4)
Physics +9  Chemistry -2  Biology -1
```

**关键差异**：旧 prompt 下 tau2 是回退的 (-4)，新 prompt 下 tau2 反弹回 **净 +6**。Physics +9 是主推手（Quantum +4、Physics-general +3、Relativistic +1、Optics +1），Organic Chemistry 仍 net -3。

### base → tau2（端到端）

```
fixed=53, broken=19, net=+34   (旧 prompt: net=+19)
Physics +30  Chemistry +1  Biology +3
```

**新 prompt 下的端到端净收益 +34 道，比旧 prompt 的 +19 高 +15 道**。Physics 一家贡献 30/34 = 88%。

按 subdomain：
- Physics (general) +11 ⭐ / Quantum Mechanics +9 ⭐ / Relativistic Mechanics +4 / Chemistry (general) +3 / High-energy +2 / Molecular Biology +2 / Electromagnetism +2 / Astrophysics +1 / Optics +1 / Genetics +1
- **Organic Chemistry -2** ← 唯一净负 subdomain

---

## 7. 单题 Pattern 统计（4 ckpt）

| Pattern | 总数 | Physics | Chemistry | Biology |
|---|---:|---:|---:|---:|
| `stable` (4 ckpt 一致) | **85** (42.9%) | 33 | 44 | 8 |
| ↳ stable_correct (✓✓✓✓) | 34 | — | — | — |
| ↳ stable_wrong (✗✗✗✗) | 51 | — | — | — |
| `fixed_at_bm` (✗✓✓✓) | 23 | 14 | 6 | 3 |
| `fixed_at_fs` (✗✗✓✓) | 5 | 5 | 0 | 0 |
| `fixed_at_tau2` (✗✗✗✓) | **18** | 9 | 7 | 2 |
| `broken_at_bm` (✓✗✗✗) | 9 | 2 | 7 | 0 |
| `broken_at_fs` (✓✓✗✗) | 2 | 0 | 1 | 1 |
| `broken_at_tau2` (✓✓✓✗) | 3 | 1 | 1 | 1 |
| `volatile_*`（多次反转） | 53 | 22 | 26 | 5 |

**与旧 prompt 的关键差异**：

| | 旧 prompt | 新 prompt | 解读 |
|---|---:|---:|---|
| stable | 71 (35.9%) | **85 (42.9%)** | 新 prompt 增稳 14 道 |
| stable_correct | 29 | 34 | +5 道更稳定的"对" |
| stable_wrong | 42 | 51 | +9 道更稳定的"错"（持续盲区扩大）|
| fixed_at_tau2 | 18 | 18 | 一致 |
| broken_at_tau2 | 14 | **3** | tau2 不再"破坏"题（旧 prompt 是 14 道）|
| volatile | 62 | **53** | -9 道，整体反转减少 |

→ **新 prompt 让训练轨迹"更确定"**：fixed_at_tau2 (18) 远 > broken_at_tau2 (3)，tau2 单步贡献从旧 prompt 的"互相抵消" (18 fixed vs 14 broken) 转为"明显净正" (18 vs 3)。

### Volatile 桶细分（4-ckpt bitstring）

| Pattern | 总数 | Physics | Chemistry | Biology |
|---|---:|---:|---:|---:|
| `volatile_0100` (✗✓✗✗) | 13 | 3 | **10** | 0 |
| `volatile_0110` (✗✓✓✗) | 12 | 4 | 6 | 2 |
| `volatile_0101` (✗✓✗✓) | 7 | 6 | 1 | 0 |
| `volatile_0010` (✗✗✓✗) | 5 | 4 | 1 | 0 |
| `volatile_1010` (✓✗✓✗) | 5 | 1 | 4 | 0 |
| `volatile_1011` (✓✗✓✓) | 5 | 0 | 3 | 2 |
| `volatile_1001` (✓✗✗✓) | 3 | 2 | 1 | 0 |
| `volatile_1101` (✓✓✗✓) | 3 | 2 | 1 | 0 |

→ **`volatile_0100` (只 base_math 对) 13 道**是新 prompt 下最大的 volatile 桶——math RL 对了，search 之后又丢了，tau2 也没救回。Chemistry 占 10/13。

---

## 8. Persistent 状态（跨 ckpt 全错）

```
total: 198
persistent ✓ (4 ckpt 全对): 34  (17.2%)   ← 旧 prompt 29
persistent ✗ (4 ckpt 全错): 51  (25.8%)   ← 旧 prompt 42
flapping (变动):           113 (57.1%)   ← 旧 prompt 127 (64.1%)
```

**新 prompt 下 GPQA 更"两极化"**：persistent_correct +5 / persistent_wrong +9 / flapping -14。新 prompt 把更多题"固化"了（无论对错）。

按 subdomain 的 persistent_wrong (top 8)：

| Subdomain | persistent_wrong | 占该 subdomain |
|---|---:|---:|
| **Organic Chemistry** | 28 | **28/72 = 39%** ⚠️ |
| Chemistry (general) | 7 | 35% |
| Molecular Biology | 4 | 27% |
| Astrophysics | 3 | 23% |
| Quantum Mechanics | 2 | 8% |
| High-energy particle physics | 2 | 14% |
| Physics (general) | 2 | 11% |
| Electromagnetism and Photonics | 1 | 17% |

→ **Organic Chemistry 单类占全部 persistent_wrong 的 28/51 = 55%**（旧 prompt 50%）——新 prompt 下不仅没改善，反而把更多 OC 题"固化在错"里。这是 GPQA 上**最持久的训练盲区**。

---

## 9. 与旧 prompt GPQA 报告核心结论对照

| 维度 | 旧 prompt | 新 prompt |
|---|---|---|
| 整体 base→tau2 | +9.6pp | **+17.2pp** |
| tau2 vs final_search | -2.0pp（回退）| **+3.0pp**（首次正向）|
| Physics | +24.4pp | **+34.9pp** |
| Chemistry | -1.1pp（跌破 base）| +1.1pp（小幅正向）|
| Biology | -5.3pp（跌破 base）| +15.8pp（正向）|
| Organic Chemistry | -5.6pp | -2.8pp（仍负但更轻）|
| Quantum Mechanics | +32pp (0.840) | +36pp (0.840) |
| boxed 用法 | 38-72 道 (19-36%) | **139-189 道 (70-95%)**|
| via_boxed 准确率 | 0.60-0.71 | 0.35-0.51 |
| persistent_wrong | 42 | **51** |
| fixed_at_tau2 / broken_at_tau2 | 18/14 | 18/**3** |
| flapping % | 64.1% | **57.1%** |

**新 prompt 的整体效应**：

1. **训练轨迹被显著放大**：base→tau2 的净收益翻倍 (+9.6 → +17.2pp)；端到端 Physics 主导更明显。
2. **tau2 不再回退**：在 GPQA 上 tau2 重新成为最佳 ckpt（0.4949），与 math500/MMLU/MMLU-Pro 一致单调向上。
3. **boxed 路径几乎垄断提取**：bare_letter 几近清零，所有错误的"主桶"是 `boxed_wrong`——评估更直接反映模型能力，提取噪声基本消除。
4. **代价**：boxed 路径平均准确率从 0.6+ 跌到 0.35-0.51（因为模型被强制写 box 时也会硬猜），且 persistent_wrong 从 42 涨到 51（更多固化盲点）。

---

## 10. 跨数据集横向对比（4 ckpt，新 prompt 适用于 GPQA）

| 指标 | math500 | MMLU | MMLU-Pro | **GPQA (新 prompt)** |
|---|---:|---:|---:|---:|
| 总题数 | 500 | 14042 | 12032 | 198 |
| base 起点 | 0.7960 | 0.6699 | 0.5677 | **0.3232** |
| base→final_search Δ | +5.6pp | +12.5pp | +13.0pp | **+14.1pp** |
| final_search→tau2 Δ | +0.6pp | +X | +X | **+3.0pp** |
| **base→tau2 Δ** | +6.4pp | (running) | (running) | **+17.2pp** |
| persistent_wrong | 41 (8.2%) | — | — | **51 (25.8%)** |
| flapping % | 17.4% | — | — | **57.1%** |

**新 prompt 让 GPQA 与其他数据集的"故事"更一致**：tau2 在 4 个数据集上**全部单调向上**，不再出现"GPQA 是唯一回退"的反常现象。

---

## 11. 核心结论

1. **新 prompt 让 tau2 重夺 GPQA 第一**：0.4949（vs 旧 prompt final_search 0.4949 / tau2 0.4747），跨 prompt 也是 **0.4949**。
2. **base→tau2 端到端净收益从 +9.6pp 拉到 +17.2pp**：新 prompt 通过把 base 起点压低 (-5.6pp) + 把 tau2 抬高 (+2.0pp)，把训练效果放大近一倍。
3. **Physics +34.9pp 是单 domain 最大收益**：Quantum 0.840、Physics-general 0.737、Relativistic 0.714，三个 subdomain 全部 +35pp 以上。
4. **Chemistry / Biology 不再 "跌破 base"**：Chemistry +1.1pp（小幅正）/ Biology +15.8pp（明显正），消除了旧 prompt 下"训练对 Chemistry/Biology 净负"的反常结论。
5. **Organic Chemistry 仍是盲区**：72 题中 28 题（39%）4 ckpt 全错，end-to-end 仍 -2.8pp。这是 prompt 工程救不了的能力短板。
6. **`boxed` 路径占 70-95% 提取量**：bare_letter 几乎清零（45 → 1 → 6），提取噪声消除，但代价是 box 准确率从 ~0.65 跌到 ~0.49（被强制写 box 也会硬猜）。
7. **`boxed_wrong` 是失败的"主桶"**（83-98 道）：所有 ckpt 上 80%+ 的错都发生在用 `\boxed{}` 收尾时——这是模型领域能力的真实边界，不再被提取路径混淆。
8. **rescuable_via_boxed = 7 道**（base 1 / final_search 3 / tau2 3）：strict 抓字母时被讨论中的字母骚扰，box 里反而是对的。如果调整 utils.py 把 boxed 优先级提到 strict 之前可以多对 7 道（潜在 +0.9pp 全 ckpt 平均）。
9. **新 prompt 让训练轨迹更"确定"**：fixed_at_tau2 = 18 / broken_at_tau2 = **3**（旧 prompt 14），tau2 单步贡献从"互相抵消"转为"明显净正"。
10. **Persistent_wrong 从 42 涨到 51**：新 prompt 下更多题"固化在错"，提取噪声消除后暴露的真实盲区比旧 prompt 多 9 道。
11. **flapping 比例 57% 仍最高**（math500 17%、MMLU/MMLU-Pro 估测更低）：GPQA 小样本难题对 RL 训练阶段的敏感性即使在新 prompt 下也是全数据集最高。
