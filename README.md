# 国债期限结构的联合风险预测与对冲有效性研究

上海财经大学金融数据挖掘课程论文 V7.1 的配套代码。研究比较注意力机制与传统动态模型，考察五期限国债收益率中的信息能否改善方向预测、风险尺度、联合风险及对冲决策。

## 运行验证

在 Windows x64 上安装 uv，从仓库根目录运行。输出目录须尚未存在。

```powershell
uv sync --locked
uv run --locked python -B smoke.py --with-tests --output .scratch/verification
```

第二条命令依次执行 165 项单元测试和五组仅用合成输入的流程检查，写出 `status.json` 与标为合成结果的示例图。只运行合成流程时可省略 `--with-tests`。运行说明和验收记录见 [验证文档](docs/verification/README.md)。

## 按论文查找代码

| 章节与图表 | 实验目录 | 主要入口与子程序 |
|---|---|---|
| §4.2–4.3，F03–F04 | [方向预测](experiments/direction_prediction/) | `experiment.py` 构造样本和训练，`models.py` 定义固定融合、均匀融合与自注意力，`analyze.py` 计算配对结果。 |
| §4.4–4.5，F05–F06 | [目标中心注意力](experiments/target_centric_attention/) | `train_target.py` 运行目标期限对照，`models_target.py` 实现 7Y 读取，`analyze_target.py` 和 `make_figures_v2.py` 计算并展示指标。 |
| §4.6–4.8，F07–F08 | [风险尺度预测](experiments/risk_scale_forecasting/) | `train_v3.py` 运行时序聚合对照，`models_v3.py` 定义模型，`numerical_v3.py` 提供 HAR 和 QLIKE，`analyze_v3.py` 与 `make_figures_v3.py` 汇总并绘图。 |
| §5.2–5.5，F09–F11 | [联合风险预测](experiments/joint_risk_forecasting/) | `train_v4.py` 组织预测，`models_v4.py` 实现矩阵检索，`financial_v4.py` 提供 DCC 等基准，`analyze_v4.py` 与 `make_figures_v4.py` 处理跨市场对照。 |
| §6.1–6.5，F12–F13 | [对冲决策](experiments/hedging_decisions/) | `hedge_v5.py` 计算对冲动作、风险及查询消融，`make_paper_v5.py` 将既有汇总结果绘成配对风险差图。 |

具体结论、测试及历史证据见 [论文证据索引](docs/PAPER_EVIDENCE.md)。原实验编号、来源路径和迁移前校验值见 [源码溯源表](docs/source-map.tsv)。

## 环境、数据与复现范围

- 根目录 `uv.lock` 锁定原环境，使用 Python 3.12 和 PyTorch 2.8.0 CUDA 12.8 wheel。便捷验证环境另有 [CPU 锁](env/cpu/uv.lock)，可使用 `uv sync --project env/cpu --locked` 安装，并在上述运行命令的 `uv run` 后加 `--project env/cpu`。CPU 依赖锁已校验，独立安装后的整套测试待验证。各依赖的精确版本见对应锁文件。
- 示例输入按市场日期排序。收益率期限顺序为 1Y、3Y、5Y、7Y、10Y，2.10 表示 2.10%，差值乘 100 转成基点。方向预测使用 20 × 5 窗口。风险尺度模型使用包含收益率变化及其平方的 20 × 10 窗口。联合风险模型另用训练期归一化和历史先验。对冲决策使用已成熟的案例历史。
- 历史入口保留在实验目录中。重算论文表格需要授权输入、原始切分、冻结清单及历史运行材料。中国和美国的样本分别评价。美国联合风险结果来自当时冻结的外部评价，对冲实验使用已曝光的数据，属于探索性证据。命令产生的合成结果用于验证程序行为。

已有授权汇总时，可运行以下命令重绘第六章的配对对照图。

```powershell
uv run --locked python experiments/hedging_decisions/make_paper_v5.py --summary LOCAL_AGGREGATE_JSON --output .scratch/hedging_pairs.png
```
