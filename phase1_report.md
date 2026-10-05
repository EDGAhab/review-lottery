# Phase 1 报告：数据管线

> 日期：2026-10-05。依据 v3.0 §7 技术路线；分析计划以 OSF 预注册为准（已提交，不偏离）。
> 本阶段只做数据构建与描述统计，未做任何模型拟合（有序 probit、决策模型、模拟均为 Phase 2）。

## 1. 管线（三个脚本，全部可复现）

| 脚本 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `scripts/01_load_clean.py` | `data/raw/iclr26v1.parquet` | `data/processed/berenslab_clean.parquet` | 加载时立即删除 `author_ids`/`authors`（作者 PII）；列名扫描确认无评审员身份字段；titles 抽样 2 万无邮箱 |
| `scripts/02_build_dataset.py` | clean parquet + `peersum_huggingface.jsonl` | `reviews.parquet`、`papers.parquet`、`peersum_reviews.parquet` | decision 归一化（15 种变体单测全过 + 50 条人工抽查）；量表断言（各年观测值集 = 预期量表）；只用 2017–2025 |
| `scripts/03_describe.py` | 三个 processed 表 | `phase1_notes/annual_table.csv`、`phase1_notes/describe.md`、`figures/*.png` | 年度描述统计 + 4 张图 + 质量门 |

## 2. 数据规模（脚本数字）

- **papers.parquet**：36,113 篇（ICLR 2017–2025）
- **reviews.parquet**：134,912 条评审（`confidence`/`review_len` 列存在但全空，berenslab 无此字段，按设计）
- **peersum_reviews.parquet**：53,732 条官方顶层评审（2018–2022，含 confidence 与评审全文长度；venue 含 ICLR + NeurIPS）
- 交叉校验通过：reviews 行数 = papers.k 求和；paper_id 集合一致；score_std 无空值。

decision 分布：accept 11,213 / reject 17,426 / unknown 7,474（unknown 全部为 withdrawn，无空值、无无法识别的变体）。

## 3. 年度描述核心数字

| 年 | 投稿 | 录用率 | 平均评审数 | 4+评审论文数 | 量表 |
|---|---|---|---|---|---|
| 2017 | 489 | 40.5% | 3.08 | 44 | 1–10 |
| 2018 | 1,012 | 33.2% | 3.01 | 19 | 1–10 |
| 2019 | 1,569 | 32.0% | 3.04 | 77 | 1–10 |
| 2020 | 2,593 | 26.5% | 3.04 | 177 | {1,3,6,8} |
| 2021 | 3,009 | 28.6% | 3.86 | 2,376 | 1–10 |
| 2022 | 3,422 | 32.0% | 3.90 | 2,676 | {1,3,5,6,8,10} |
| 2023 | 4,955 | 31.8% | 3.78 | 3,414 | {1,3,5,6,8,10} |
| 2024 | 7,401 | 30.6% | 3.86 | 5,432 | {1,3,5,6,8,10} |
| 2025 | 11,663 | 31.8% | 4.06 | 9,780 | {1,3,5,6,8,10} |

- 投稿量 9 年约 24 倍；录用率稳定在 26–41%。
- 4+ 评审论文合计 **23,995 篇**（2021–2025 占 23,678），与 Phase 0 报告数字**完全一致**，内部锚点样本充足。
- 分数均值：除 2020 年（粗量表，均值 4.18）外各年 5.15–5.69。
- PeerSum ICLR 覆盖：2018–2022 每年 2,748–10,441 条评审，confidence 完整率 100%，评审平均长度 2,316–3,224 字符；2023–2025 无 confidence/全文（预注册已声明 RQ-C 字段基础收窄）。

## 4. 质量门结论

触发标记（已调查，不擅自删年份，待判定）：

- **2018–2025 decision 缺失率 > 5%**（9.4% → 25.3%，逐年上升）。调查：**缺失全部为 withdrawn**（7,474 篇），无空值、无无法识别的 decision 变体。withdrawn 论文多数有评审分数（仅 496 篇无分数）。
- 每年评审数中位数 ≥ 3（3.0–4.0），此门未触发。

**对 Phase 2 的影响（建议）**：
1. withdrawn 论文应纳入 RQ-A 方差分解（有分数，贡献 θ 估计），但排除在决策模型拟合之外（无 outcome）。这是标准做法，不违反预注册。
2. 撤稿率逐年上升至 25% 本身是一个值得在论文讨论部分提及的现象（可能与投稿量膨胀/多投有关），但不进入主分析。
3. 不删除任何年份。2026（19,793 篇，无分数无决定）已按计划排除，待放榜后补。

## 5. 出口标准判定

v3.0 Phase 1 出口：`reviews.parquet` + `papers.parquet` + 年度描述表。

- [x] `data/processed/reviews.parquet`（134,912 行，10 列）
- [x] `data/processed/papers.parquet`（36,113 行，10 列）
- [x] `data/processed/peersum_reviews.parquet`（53,732 行，RQ-C 补充）
- [x] `phase1_notes/annual_table.csv` + `phase1_notes/describe.md`
- [x] `figures/` 4 张图（投稿数、录用率、分数分布、每篇评审数）

**结论：Phase 1 达标，可进入 Phase 2**（有序 probit 方差分解、决策模型、双委员会模拟、双重校准）。Phase 2 启动前请确认质量门第 4 节的 withdrawn 处理建议。

## 6. 文件清单

- 脚本：`scripts/01_load_clean.py`、`scripts/02_build_dataset.py`、`scripts/03_describe.py`
- 数据：`data/processed/berenslab_clean.parquet`、`reviews.parquet`、`papers.parquet`、`peersum_reviews.parquet`（原始 `data/raw/` 未动，已 gitignore）
- 笔记：`phase1_notes/annual_table.csv`、`phase1_notes/describe.md`
- 图：`figures/fig_submissions_by_year.png`、`fig_accept_rate_by_year.png`、`fig_score_dist_by_year.png`、`fig_reviews_per_paper.png`
- 本报告：`phase1_report.md`
