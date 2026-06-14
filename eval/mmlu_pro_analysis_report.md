# MMLU-Pro 详细分析报告（base / base_math / final_search / tau2）

**版本**：v3 — 加入 tau2，提取逻辑 3 层（strict → boxed → bare）

脚本：
- [tools/eval_analysis/classify_mmlu_pro_failures.py](../tools/eval_analysis/classify_mmlu_pro_failures.py)
- [tools/eval_analysis/breakdown_mmlu_pro_eval.py](../tools/eval_analysis/breakdown_mmlu_pro_eval.py)
- [tools/eval_analysis/regrade_mmlu_pro_jsonl.py](../tools/eval_analysis/regrade_mmlu_pro_jsonl.py)

CSV 导出：`/tmp/mmlu_pro_per_doc_4ckpt.csv`（12032 行）

> MMLU-Pro：12032 题，14 个 native category，10 选项 (A-J)。提取顺序：
> 1. **strict** — `the answer is (X)` 官方模板
> 2. **boxed** — `\boxed{X}`（math RL 训练后的格式迁移）
> 3. **bare** — 末尾最后一个 `\b[A-J]\b` 字母 兜底
> 4. **no_match** — 完全没字母

---

## 1. 总体准确率（4 ckpt）

| | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|
| 原始 exact_match | 0.5677 | 0.6516 | 0.6971 | **0.7255** | **+15.78pp** |
| re-live 提取 | 0.5677 | 0.6516 | 0.6971 | 0.7255 | （完全一致） |

**关键发现**：tau2 是 4 ckpt 最高（**0.7255**），比 final_search 净 +2.84pp，端到端 +15.78pp。MMLU-Pro 是 tau2 vs final_search 涨幅第二大的数据集（仅次于 MMLU 的 +3.36pp）。

---

## 2. Layer 1 — 提取路径分布（揭示训练对输出格式的塑造）

| 提取路径 | base | base_math | final_search | **tau2** |
|---|---:|---:|---:|---:|
| `strict` (`the answer is (X)`) | 10694 (88.9%) | 7672 (63.8%) | 7545 (62.7%) | **11284 (93.8%)** |
| `boxed` (`\boxed{X}`) | 130 (1.1%) | **3399 (28.2%)** | **3283 (27.3%)** | **163 (1.4%)** |
| `bare` (散装末尾字母) | 1171 (9.7%) | 945 (7.9%) | 1198 (10.0%) | **585 (4.9%)** |
| `no_match` | 37 (0.3%) | 16 (0.1%) | 6 (0.0%) | **0 (0.0%)** |

**核心发现**（跟 MMLU 一致的"主动卸下 boxed"现象）：
1. **tau2 的 strict 用法 93.8%，比 base 的 88.9% 还高**：模型不只是回归 base 风格，而是**比 base 更严格遵守官方模板**
2. **`boxed` 用法戏剧性回落**：base 130 → base_math 3399 → final_search 3283 → tau2 **163**，几乎完全消除了 math RL 带来的格式漂移
3. **`bare` 创新低 585 (4.9%)**：之前最少也是 945，tau2 把"散装字母"压到不到 5%
4. **`no_match` 彻底清零 (0/12032)**：MMLU-Pro 上**没有一个样本提取不到字母**

### 各路径准确率（每条路径有多准）

| 路径 | base acc | base_math acc | final_search acc | tau2 acc |
|---|---:|---:|---:|---:|
| `strict` | 6441/10694 (0.602) | 4974/7672 (0.648) | 5249/7545 (0.696) | **8180/11284 (0.725)** |
| `boxed` | 94/130 (0.723) | 2583/3399 (0.760) | 2544/3283 (0.775) | **141/163 (0.865)** |
| `bare` | 295/1171 (0.252) | 283/945 (0.300) | 595/1198 (0.497) | **408/585 (0.697)** |
| 整体 | 0.568 | 0.652 | 0.697 | **0.725** |

**关键洞察**：
- **每条路径的准确率都单调上升**——不是格式选择变化驱动的"虚假提升"，是模型推理本身在变好
- **`boxed` 路径准确率从 0.775 跳到 0.865**：tau2 上模型用 \boxed{} 时是 4 ckpt 中最有把握的（虽然只 163 道走 boxed）
- **`bare` 路径准确率从 0.497 跳到 0.697**：之前 bare 是低质信号，tau2 上变得跟整体准确率几乎一样可靠
- **`strict` 路径准确率 0.725** 等于整体准确率：strict 是"主流路径"，跟整体能力同步

---

## 3. Layer 2 — no_match 内部审计

| | base | base_math | final_search | tau2 |
|---|---:|---:|---:|---:|
| no_match 总数 | 37 | 16 | 6 | **0** |
| `hidden_correct` | 0 | 0 | 0 | 0 |
| `hidden_failure` | 0 | 0 | 0 | 0 |
| 真的没字母 | 37 | 16 | 6 | 0 |

tau2 完全消除了"输出里完全没字母"的样本，这意味着模型在每个 MMLU-Pro 题上都至少给了一个 A-J 字母。

---

## 4. Layer 3 — 失败子类（4 ckpt）

| 子类 | base | base_math | final_search | **tau2** | 备注 |
|---|---:|---:|---:|---:|---|
| `extracted_wrong` (strict 答错) | 4253 | 2698 | 2296 | **3104** | strict 路径用得多，错的总数也跟着多 |
| `boxed_wrong` (boxed 答错) | 36 | 816 | 739 | **22** | boxed 用量降，错也降 |
| `fallback_wrong` (bare 答错) | 876 | 662 | 603 | **177** | bare 准确率提升+用量降，双重原因导致大幅下降 |
| `gave_up_short` | 23 | 5 | 1 | **0** | 完全清零 |
| `medium_unconverged` | 11 | 8 | 2 | **0** | 完全清零 |
| `repetition_loop` | 3 | 2 | 1 | 0 | |
| `truncated_max_tokens` | 0 | 0 | 2 | 0 | |
| `python_block_unfinished` | 0 | 1 | 0 | 0 | |

**关键观察**：
- **格式相关失败完全清零**：tau2 上 `gave_up_short` / `medium_unconverged` / `repetition_loop` / `truncated_max_tokens` 全 0
- **`fallback_wrong` 大幅下降** (603 → 177)：bare 路径用量减半 + 准确率从 0.50 跳到 0.70 的双重作用
- **`extracted_wrong` 反而上升** (2296 → 3104)：因为 strict 路径承担了更多样本（7545 → 11284），承担越多错误数也越多——但 strict 路径的"错误率"实际上是降低的（30.4% → 27.5%）

---

## 5. 按 14 个 category（按 final_search 准确率排序）

| Category | n | base | base_math | final_search | tau2 | base→tau2 Δ |
|---|---:|---:|---:|---:|---:|---:|
| **math** | 1351 | 0.6373 | 0.7898 | 0.8327 | **0.8890** | **+0.2517** ⭐ |
| **physics** | 1299 | 0.5820 | 0.7044 | 0.7475 | **0.7875** | **+0.2055** |
| **chemistry** | 1132 | 0.5777 | 0.7235 | 0.7359 | **0.7809** | **+0.2032** |
| **business** | 789 | 0.5856 | 0.7009 | 0.7567 | **0.7820** | **+0.1965** |
| **engineering** | 969 | 0.4314 | 0.5439 | 0.5748 | **0.6171** | **+0.1858** |
| biology | 717 | 0.7252 | 0.7992 | 0.8424 | **0.8591** | +0.1339 |
| computer science | 410 | 0.6171 | 0.6927 | 0.7415 | **0.7585** | +0.1415 |
| other | 924 | 0.5325 | 0.6061 | 0.6504 | **0.6851** | +0.1526 |
| economics | 844 | 0.6517 | 0.7358 | 0.7867 | **0.8021** | +0.1505 |
| psychology | 798 | 0.6679 | 0.6867 | 0.7306 | 0.7506 | +0.0827 |
| philosophy | 499 | 0.5331 | 0.5391 | 0.5812 | 0.6192 | +0.0862 |
| health | 818 | 0.6222 | 0.6577 | 0.7078 | 0.7115 | +0.0892 |
| **law** | 1101 | 0.3252 | 0.3306 | 0.4124 | **0.4124** | +0.0872 ⚠️ |
| history | 381 | 0.5197 | 0.5302 | 0.5932 | 0.5906 | +0.0709 |

**关键观察**：
- **math 涨幅最大** (+25.17pp)：base 0.637 → tau2 **0.889** —— MMLU-Pro 数学题接近天花板
- **STEM 三大类 (math/physics/chemistry) 全都 +20pp 以上**
- **business 涨幅 +19.65pp**：超过了 STEM 的 chemistry，是非 STEM 中表现最好的
- **law 在 tau2 阶段完全不动**（0.4124 → 0.4124，fixed=144 / broken=144 完全相抵）—— 这是 MMLU-Pro 上 tau2 唯一"零进步" category，但端到端仍 +8.72pp 是有提升的
- **history 微跌**（0.5932 → 0.5906，-1 道）：唯一净负 category，但样本小 (381) 且只差 1 道
- **没有任何 category 在端到端上回退**

---

## 6. Fixed / Broken 矩阵（按 category）

### 6.1 base → base_math（math RL）
**总：fixed=1961, broken=951, net=+1010**

按 category top 5 净增益：math +206, chemistry +165, physics +159, engineering +109, business +91

### 6.2 base_math → final_search（search 训练）
**总：fixed=1348, broken=800, net=+548**

按 category top 5 净增益：law **+90**, math +58, physics +56, business +44, economics +43

→ search 阶段 law 净增最大（这是之前报告里的关键发现）

### 6.3 final_search → tau2（最新阶段）

| Category | fixed | broken | net |
|---|---:|---:|---:|
| **math** | 121 | 45 | **+76** |
| physics | 141 | 89 | +52 |
| chemistry | 106 | 55 | +51 |
| engineering | 134 | 93 | +41 |
| **other** | 85 | 53 | +32 |
| business | 63 | 43 | +20 |
| philosophy | 56 | 37 | +19 |
| psychology | 53 | 37 | +16 |
| economics | 55 | 42 | +13 |
| biology | 44 | 32 | +12 |
| computer science | 42 | 35 | +7 |
| health | 53 | 50 | +3 |
| **law** | 144 | 144 | **±0** ⚠️ |
| history | 25 | 26 | **-1** |
| **总计** | **1122** | **781** | **+341** |

→ tau2 阶段 STEM 类 (math+physics+chemistry+engineering) 净 +220 占总净增的 65%。**law 的"零进步"** 是 tau2 训练的明确边界——之前 search 训练在 law 上有 +90 净增，tau2 没继续，但也没退步。

### 6.4 base → tau2（端到端）
**总：fixed=2532, broken=633, net=+1899**

| Category | fixed | broken | net |
|---|---:|---:|---:|
| **math** | 369 | 29 | **+340** |
| physics | 337 | 70 | +267 |
| chemistry | 277 | 47 | +230 |
| engineering | 259 | 79 | +180 |
| business | 186 | 31 | +155 |
| other | 189 | 48 | +141 |
| economics | 169 | 42 | +127 |
| **law** | 190 | 94 | +96 |
| biology | 130 | 34 | +96 |
| health | 124 | 51 | +73 |
| psychology | 95 | 29 | +66 |
| computer science | 77 | 19 | +58 |
| philosophy | 77 | 34 | +43 |
| history | 53 | 26 | +27 |

→ 端到端 fixed:broken = **4:1**，比 base→final_search 的 3.1:1 还干净。math 一家净 +340，占总净增的 18%。

---

## 7. 单题 Pattern 统计（4 ckpt）

| Pattern | 总数 | Top 4 categories |
|---|---:|---|
| `stable` (4 ckpt 一致) | **7286** (60.6%) | math(829), physics(726), chemistry(690), law(588) |
| `fixed_at_bm` (✗→✓→✓→✓) | 1363 | math(232), physics(197), chemistry(163), engineering(118) |
| `fixed_at_fs` (✗→✗→✓→✓) | 469 | math(54), chemistry(52), engineering(51), law(48) |
| **`fixed_at_tau2`** (✗→✗→✗→✓) | **486** | **law(68)**, physics(62), engineering(60), math(57) |
| `broken_at_bm` (✓→✗→✗→✗) | 204 | engineering(31), law(27), physics(21), chemistry(19) |
| `broken_at_fs` (✓→✓→✗→✗) | 110 | law(19), engineering(14), physics(12), other(9) |
| **`broken_at_tau2`** (✓→✓→✓→✗) | **215** | physics(28), law(27), health(21), engineering(20) |
| `volatile_1011` (✓→✗→✓→✓) | **489** | physics(56), law(47), math(45), other(44) |
| `volatile_1101` (✓→✓→✗→✓) | 268 | law(38), chemistry(30), physics(28), other(22) |
| `volatile_0010` (✗→✗→✓→✗) | 286 | **law(81)**, engineering(31), physics(28), chemistry(18) |
| `volatile_0101` (✗→✓→✗→✓) | 214 | physics(32), engineering(30), math(26), chemistry(24) |
| `volatile_0100` (✗→✓→✗→✗) | 208 | engineering(36), law(30), chemistry(19), other(19) |
| `volatile_0110` (✗→✓→✓→✗) | 176 | engineering(28), physics(24), chemistry(17), law(15) |
| `volatile_1001` (✓→✗→✗→✓) | 154 | engineering(25), physics(19), law(18), math(18) |
| `volatile_1010` (✓→✗→✓→✗) | 104 | law(21), engineering(14), biology(9), other(9) |

**关键观察**：
- **`fixed_at_tau2` 486 道，law 占 68（14%）**：tau2 在 law 上修了一些之前都搞不定的
- **`volatile_0010` 里 law 占 81 道（28%）**：这是"final_search 短暂修好但 tau2 又破坏"的题——tau2 上 law 的"零净增"正好对应这个反复
- **`volatile_1011` 489 道**：base 对 → bm 错 → fs 对 → tau2 对（base_math 阶段独有的"中间塌陷"）—— math RL 短暂破坏了不少题
- **law 在每个 volatile pattern 都占大头**：MMLU-Pro 上 law 是最不稳定的 category

---

## 8. Persistent 状态

```
total: 12032
persistent ✓ (4 ckpt 全对): 5286  (43.9%)
persistent ✗ (4 ckpt 全错): 2000  (16.6%)   ← 比 3-ckpt 的 2486 减少 486 道！
flapping (变动):           4746  (39.5%)
```

**persistent_wrong 减少 486 道** —— tau2 拿下了之前三 ckpt 都没解决的接近 20% 硬题。

---

## 9. 与 3-ckpt 报告的核心结论变化

| 维度 | 3-ckpt 结论 | 加上 tau2 后 |
|---|---|---|
| 整体趋势 | base→final 单调上升 (+13pp) | **tau2 继续 +2.84pp，达到 0.7255** |
| `boxed` 用法 | math RL 飙升到 28%，search 持平 | **tau2 完全卸下回到 1.4%**（接近 base 1.1%）|
| `strict` 用法 | math RL 后跌到 64%，没回升 | **tau2 飙到 93.8%，比 base 88.9% 还高** |
| `boxed` 准确率 | 0.72→0.76→0.78 | **tau2 0.865，最高** |
| `bare` 准确率 | 0.25-0.50 之间，低质信号 | **tau2 0.697，跟整体准确率几乎一致** |
| `no_match` 演化 | base 37 → final 6 | **tau2 0，彻底清零** |
| math 涨幅 | +19.5pp | **+25.17pp** (端到端，base→tau2) |
| law 趋势 | search 阶段 +9pp 是搜索训练价值的最强信号 | **tau2 阶段 ±0**——law 的训练边界 |

---

## 10. 跨数据集横向对比（4 ckpt）

| 指标 | math500 | MMLU | **MMLU-Pro** | GPQA |
|---|---:|---:|---:|---:|
| 总题数 | 500 | 14042 | **12032** | 198 |
| base 起点 | 0.7960 | 0.6699 | **0.5677** | 0.3788 |
| base→final_search Δ | +5.6pp | +12.5pp | +13.0pp | +11.6pp |
| final_search→tau2 Δ | +0.6pp | **+3.4pp** ⭐ | **+2.8pp** ⭐ | -2.0pp |
| **base→tau2 Δ** | +6.4pp | **+15.8pp** | **+15.8pp** | +9.6pp |
| persistent_wrong | 41 (8.2%) | 1215 (8.7%) | **2000 (16.6%)** | 42 (21.2%) |
| flapping % | 17.4% | 38.2% | **39.5%** | 64.1% |
| boxed 用法 (final_search) | N/A | 7.2% | **27.3%** | 20.2% (via_boxed) |
| boxed 用法 (tau2) | N/A | 0.3% | **1.4%** | 19.2% (via_boxed) |

**MMLU 和 MMLU-Pro 在 tau2 阶段表现高度一致**：
- 都涨 +2.8-3.4pp
- 都让 boxed 用法戏剧性回落（27%/7% → 1.4%/0.3%）
- 都让 strict 用法飙升回 90%+
- 都让 no_match 接近清零

GPQA 是唯一在 tau2 阶段回退的数据集（-2pp），且 boxed 用法基本不变（20.2% → 19.2%）。

---

## 11. 核心结论

1. **tau2 是 MMLU-Pro 上最佳 ckpt**（0.7255），比 final_search 净 +2.84pp，端到端 +15.78pp（与 MMLU 完全一致的端到端提升）
2. **跟 MMLU 完全平行的"格式回归"现象**：
   - `\boxed{X}` 用量从 final_search 3283 道（27.3%）压回 **163 道（1.4%）**
   - `the answer is (X)` 用量从 final_search 7545 飙到 **11284**（93.8%，比 base 88.9% 还高）
   - `no_match` 彻底清零（最严格 prompt 也都遵守）
3. **每条提取路径的准确率都在升**：
   - strict 0.602 → 0.725
   - boxed 0.723 → 0.865
   - bare 0.252 → 0.697
   不是"格式选择更好导致虚假提升"，而是**模型推理本身变强**
4. **STEM 三大类继续上升**：math 0.889、physics 0.787、chemistry 0.781——math 接近天花板
5. **law 在 tau2 阶段完全停滞**（0.4124 → 0.4124，fixed=broken=144）—— 这是 MMLU-Pro 上 tau2 训练的明确边界。但 history -1 道是唯一净负 category（噪声级别）
6. **persistent_wrong 减少 486 道**（2486 → 2000），tau2 拿下了三 ckpt 都没解决的近 20% 硬题
7. **法律仍是 MMLU-Pro 上最难的 category**（base 0.325 → tau2 0.412，绝对值最低），且在每个 volatile pattern 都占大头——是 MMLU-Pro 训练最不稳定的领域
8. **fixed:broken = 4:1（端到端最干净）**：base→tau2 共 2532 道修好、633 道破坏，比 base→final_search 的 3.1:1 还好——tau2 训练的"信噪比"是所有 transition 中最高的
