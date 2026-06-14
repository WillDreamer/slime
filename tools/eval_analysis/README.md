# Eval Analysis Tools

后置评估分析工具集，配套 `slime/eval/results_*` 下面 lm-eval 跑出来的 `samples_*.jsonl`。
所有脚本互相独立，按数据集 × 任务类型分组：

## 脚本一览

| 脚本 | 用途 | 输入 |
|---|---|---|
| `analyze_math_eval.py` | math 单/多 ckpt 概览（旧版，含 sympy regrade 选项） | `samples_math500_slime_*.jsonl` |
| `regrade_math_jsonl.py` | math 用 sympy fallback 重打分，写回 jsonl | `samples_math500_slime_*.jsonl` |
| `regrade_mmlu_jsonl.py` | MMLU 用 broader 提取重打分（写 `extraction_method`/`extracted_letter`） | `samples_mmlu_slime_*.jsonl` |
| `regrade_mmlu_pro_jsonl.py` | MMLU-Pro 用 3 层提取（strict/boxed/bare）重打分 | `samples_mmlu_pro_slime_*.jsonl` |
| `classify_math_failures.py` | math 三层失败审计（hidden_correct/failure + 子类） | `samples_math500_slime_*.jsonl` |
| `classify_mmlu_failures.py` | MMLU 三层失败审计 | `samples_mmlu_slime_*.jsonl` |
| `classify_mmlu_pro_failures.py` | MMLU-Pro 三层失败审计（live 提取 + bare-letter fallback path） | `samples_mmlu_pro_slime_*.jsonl` |
| `classify_gpqa_failures.py` | GPQA 三层失败审计 + via_boxed 跟踪 + boxed 提取改进审计 | `samples_gpqa_slime_*.jsonl` |
| `breakdown_math_eval.py` | math 难度 × 题型矩阵 + fixed/broken | `samples_math500_slime_*.jsonl` |
| `breakdown_mmlu_eval.py` | MMLU category × subject 矩阵 + fixed/broken | `samples_mmlu_slime_*.jsonl` |
| `breakdown_mmlu_pro_eval.py` | MMLU-Pro 14 类 + fixed/broken | `samples_mmlu_pro_slime_*.jsonl` |
| `breakdown_gpqa_eval.py` | GPQA 3 domain × 13 subdomain + fixed/broken + via_boxed 跟踪 | `samples_gpqa_slime_*.jsonl` |
| `_mmlu_subjects.py` | MMLU 57→4 类别映射（被 breakdown_mmlu_eval 引用） | — |

## 共同 CLI 约定

- 所有 `classify_*` 和 `breakdown_*` 脚本都接受 `NAME=PATH` 形式的多 ckpt 输入：
  ```bash
  python classify_math_failures.py \
      base=/path/.../samples_*.jsonl \
      base_math=/path/.../samples_*.jsonl \
      final_search=/path/.../samples_*.jsonl
  ```
- `breakdown_*` 都支持 `--csv out.csv` 导出每题级 CSV
- `classify_*` 都支持 `--dump-dir out/` 把每个失败子类的样本写成 txt 方便人工审

## 已写好的报告（在 `eval/`）

- [eval/math500_analysis_report.md](../../eval/math500_analysis_report.md) — 4 ckpt (含 tau2)
- [eval/mmlu_analysis_report.md](../../eval/mmlu_analysis_report.md)
- [eval/mmlu_pro_analysis_report.md](../../eval/mmlu_pro_analysis_report.md)
- [eval/gpqa_analysis_report.md](../../eval/gpqa_analysis_report.md)

## 必需依赖

`classify_math_failures.py` 和 `regrade_math_jsonl.py` 需要 sympy + pylatexenc：
```
/xuanwu-tank/north/xw27/envs/sglang_env/bin/python  # 已装好这两个 + lm_eval
```
其他脚本只用标准库。
