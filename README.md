# 国债期限结构与风险预测｜课程论文 V7.1 配套代码

本仓库面向获准访问的课程复核者。收录 V1–V5 课程研究独立实现与合成检验。论文以 V7.1 定稿为准。

## 环境与入口

原研究环境锁位于根目录，使用 Windows x64、Python 3.12 和 PyTorch 2.8.0 CUDA 12.8 wheel。便捷验收另用 `env/cpu/` 的独立 uv 锁，使用 Python 3.12 和同版本 CPU wheel，无需 NVIDIA 驱动、CUDA 或市场数据。CPU 锁已校验，本机 CPU 安装后的整套复测待完成。下列命令从仓库根目录运行。

```powershell
uv sync --project env/cpu --locked
foreach ($round in 1..5) {
    uv run --project env/cpu --locked python -B -m unittest discover -s "v$round" -v
    if ($LASTEXITCODE -ne 0) { throw "Tests failed for v$round" }
}
uv run --project env/cpu --locked python -B smoke.py --output .scratch/synthetic
```

如使用根目录原研究依赖，执行 `uv sync --locked`，并移除以上 `uv run` 的 `--project env/cpu`。CPU 路径验证机制、数值、合成统计和绘图。真实历史复跑还需要授权输入及冻结材料。

合成入口需要一个尚不存在的输出目录。输出包含测试状态、绘图夹具和标明合成证据的示意图。测试覆盖历史实现的关键数值行为，真实训练和旧账本均不在入口调用范围内。

## 论文与代码对应

[逐项结论、消融、测试及真实复现条件](docs/PAPER_EVIDENCE.md) · [发布验收记录](docs/verification/README.md)

| 论文证据 | 代码 | 校验路径 |
|---|---|---|
| 方向预测与跨期限读取 | `v1/models.py`、`v2/models_target.py` | Self-Attention、固定及均匀融合、7Y Query、对照初始化、方向分母和配对比较 |
| 风险尺度与历史时间聚合 | `v3/models_v3.py`、`v3/numerical_v3.py`、`v3/analyze_v3.py` | 固定、均匀与 Content；HAR、标量 QLIKE、连续时间块推断 |
| 联合风险与跨市场比较 | `v4/models_v4.py`、`v4/financial_v4.py`、`v4/analyze_v4.py` | 固定核、EWMA、均匀融合、DCC、HAR-CCC、GARCH-CCC、矩阵 QLIKE、同日配对及多重比较 |
| 风险决策与查询消融 | `v5/hedge_v5.py`、`v5/make_paper_v5.py` | 对冲风险、Content/NoQuery、记忆对照与配对风险差图 |

各轮的分析代码保留原评价范围。V4 美国结果属于当时冻结的外部评价，V5 复用了已曝光的 V4 评价数据，属于探索性证据。

## 数据接口及复现范围

- V1/V2 方向模型接受按时间顺序的 20 × 5 窗口，期限顺序为 1Y、3Y、5Y、7Y、10Y。训练折决定变换，标签成熟后方可进入拟合。状态模型的附加状态接口见 V1 模型。
- V3 时间模型接受 20 × 10 的训练期标准化输入，包含五期限变化及其平方。五日平方变化聚合是风险尺度代理。HAR 的设计矩阵和正值约束见数值模块。
- V4 矩阵模块接受训练期 RMS 标准化的 20 × 5 float64 输入与 3 × 3 历史先验。评估对象为五期第二矩矩阵。中国源为活跃券收益率序列，美国源为财政部标准期限指示性曲线。两种口径的差异应随结果解释。
- V5 核心计算接受三维风险矩阵与已成熟的案例记忆。冻结回放依赖获准的 V4 本地预测及清单。仓库中的回放加载函数会明确拒绝缺少授权身份材料的历史运行。

仓库可以验证算法、数值接口、消融、合成统计和绘图路径。逐日级或逐位复算论文数字需要经授权的原始快照、完整冻结协议、旧闸门、签名清单和相关运行材料，须在原本地研究环境执行。仓库中的冻结配置为剥离来源身份和路径的参数快照，不充当授权闸门，也不用于开始新的历史拟合。已有论文数值以定稿为准。

[源码溯源与发布边界](docs/RELEASE_AUDIT.md)

## V7.1 图表对照

- 第 4 章 F03–F08 对应 V1–V3 的机制、训练和消融。`v1/analyze.py`、`v2/make_figures_v2.py` 和 `v3/make_figures_v3.py` 提供各轮原始分析及绘图路径。
- 第 5 章 F09–F11 对应 V4 风险矩阵及跨市场对照。`v4/make_figures_v4.py` 从授权本地分析目录读取聚合指标、区间和训练轨迹。
- 第 6 章 F12–F13 对应 V5 对冲风险与查询消融。可用 `uv run --locked python v5/make_paper_v5.py --summary LOCAL_AGGREGATE_JSON --output .scratch/v5_pairs.png` 从已授权汇总重绘对照图。
- 论文中的示意图、综合图及重排版使用的定稿图只随论文包交付。这里的各轮绘图入口用于复核数据到图形的处理过程，输出布局与 V7.1 定稿可能不同。

通用示例字段为 `date,yield_1y,yield_3y,yield_5y,yield_7y,yield_10y`，收益率按百分数值输入。例如 2.10 表示 2.10%，差分到基点需乘 100。示例命名与公司本地接口分离。
