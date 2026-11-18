"""Draw publication figures from saved V4 summaries and training traces."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "deliverables" / "submission" / "figures"
MODELS = ["Prior", "Uniform", "EWMA", "Kernel", "HAR_CCC", "GARCH_CCC",
          "DCC", "Static", "MLP", "Content"]
COMPARISONS = ["Static", "MLP", "Kernel", "EWMA", "HAR_CCC", "GARCH_CCC", "DCC"]
FINANCIAL = ["HAR_CCC", "GARCH_CCC", "DCC"]
SCORERS = ["Static", "MLP", "Content"]
FOLDS = {"exploration": {"2022", "2023", "2024"}, "confirmation": {"US"}}
SEEDS = {2026110541, 2026110542, 2026110543}
SUBTITLE = {
    "exploration": "中国探索样本 · 2022、2023 与 2024 年上半年等权",
    "confirmation": "美国外部确认 · 三个预定两年子期等权",
}
PALETTE = {
    "fixed": "#566478", "financial": "#237667", "Static": "#315884",
    "MLP": "#725293", "Content": "#AA441F", "axis": "#252F3A", "grid": "#E0E5EA",
}


def setup_style():
    font = font_manager.FontProperties(fname=Path("C:/Windows/Fonts/msyh.ttc"))
    font_manager.fontManager.addfont(font.get_file())
    plt.rcParams.update({
        "font.family": [font.get_name(), "DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.6, "ytick.labelsize": 8.6, "text.color": PALETTE["axis"],
        "axes.labelcolor": PALETTE["axis"], "xtick.color": PALETTE["axis"],
        "ytick.color": PALETTE["axis"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#82909D", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
    })


def heading(fig, title, subtitle, footer):
    fig.text(.045, .965, title, fontsize=13, fontweight="bold", va="top")
    fig.text(.045, .900, subtitle, fontsize=8.8, va="top")
    fig.text(.045, .031, footer, fontsize=8.2, va="bottom", linespacing=1.35)


def style_axes(ax, horizontal=True):
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color=PALETTE["grid"], linewidth=.65)
    if horizontal:
        ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3)


def save(fig, output):
    fig.savefig(output)
    plt.close(fig)
    print(output)


def results(analysis, output, phase):
    data = pd.read_csv(analysis / "model_metrics.csv")
    data = data.loc[data.aggregation.eq("macro")].set_index("model", verify_integrity=True)
    if set(data.index) != set(MODELS):
        raise ValueError("Expected all ten frozen models in the macro summary")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.45, 4.48))
    fig.subplots_adjust(left=.185, right=.95, top=.79, bottom=.14, wspace=.31)
    heading(fig, "十种方法的样本外风险预测", SUBTITLE[phase],
            "两项指标均越小越好。对冲指标为利率变动能量代理，单位为 bp²。")
    colors = [PALETTE.get(name, PALETTE["financial"] if name in FINANCIAL
                          else PALETTE["fixed"]) for name in MODELS]
    y = np.arange(len(MODELS))
    for ax, column, label, tick in zip(
        axes, ["full_qlike", "hedge_risk"], ["每维矩阵 QLIKE", "对冲后能量（bp²）"], [.3, .3]
    ):
        values = data.loc[MODELS, column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite macro score in {column}")
        span = np.ptp(values)
        ax.set_xlim(values.min() - .11 * span, values.max() + .28 * span)
        ax.scatter(values, y, c=colors, s=24, zorder=3)
        for i, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (value, i), xytext=(5, 0),
                        textcoords="offset points", fontsize=8.0, va="center")
        ax.set_yticks(y, [model.replace("_", "-") for model in MODELS])
        ax.set_title(label, loc="left", pad=10)
        ax.xaxis.set_major_locator(MultipleLocator(tick))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        style_axes(ax)
    axes[0].invert_yaxis()
    save(fig, output)


def read_intervals(analysis):
    data = pd.read_csv(analysis / "paired_intervals.csv")
    data = data.loc[data.model.eq("Content") & data.statistic.eq("macro") & data.block.eq(10)]
    expected = {(family, base) for family in ["full", "shape"] for base in COMPARISONS}
    expected.update(("hedge", base) for base in FINANCIAL)
    data = data.set_index(["family", "baseline"], verify_integrity=True)
    columns = ["estimate", "lower_95", "upper_95", "family_upper"]
    if set(data.index) != expected or not np.isfinite(data[columns]).all().all():
        raise ValueError("Missing or nonfinite primary paired inference")
    return data[columns]


def intervals(ax, data, names, point_gate=None):
    y = np.arange(len(names))
    ax.set_yticks(y, [name.replace("_", "-") for name in names])
    ax.set_ylim(len(names) - .45, -.55)
    ax.axvline(0, color="#647180", linewidth=.95, zorder=1)
    if point_gate is not None:
        ax.axvline(point_gate, color=PALETTE["Content"], linestyle=":", linewidth=1.2, zorder=2)
    ax.hlines(y, data.lower_95, data.upper_95, color=PALETTE["Static"], linewidth=1.25, zorder=2)
    ax.scatter(data.estimate, y, color=PALETTE["Static"], s=21, zorder=3, label="点估计")
    ax.scatter(data.family_upper, y - .15, facecolor="white", edgecolor=PALETTE["Content"],
               linewidth=1.0, s=21, marker="D", zorder=4, label="族内调整单侧上界")
    lower = min(data.lower_95.min(), 0, point_gate or 0)
    upper = max(data.upper_95.max(), data.family_upper.max(), 0)
    pad = .07 * (upper - lower)
    ax.set_xlim(lower - pad, upper + pad)
    style_axes(ax)


def pairs(data, output, phase):
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 5.53), width_ratios=[1.18, 1])
    fig.subplots_adjust(left=.185, right=.95, top=.815, bottom=.18, hspace=.62, wspace=.72)
    heading(fig, "Content 与七个对照的配对矩阵损失差", SUBTITLE[phase] + " · 主块长 10",
            "圆点为点估计，横线为名义 95% 区间，空心菱形为族内调整单侧上界。\n"
            "差值为 Content − 对照，负值有利于 Content。橙色点线为全矩阵事前点值门 −0.01。")
    for row, family in enumerate(["full", "shape"]):
        group = data.loc[family]
        name = "全矩阵" if family == "full" else "共同对角"
        gate = -.01 if family == "full" else None
        detail = ["Static", "MLP", "Kernel", "EWMA"]
        if family == "shape":
            detail.append("DCC")
        for col, names in enumerate([COMPARISONS, detail]):
            ax = axes[row, col]
            ax.set_title(f"{name} · " + ("全部" if col == 0 else "局部放大"), loc="left", pad=7)
            intervals(ax, group.loc[names], names, gate)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f" if col == 0 else "%.3f"))
    fig.text(.57, .113, "每维矩阵 QLIKE 差", ha="center", fontsize=9)
    save(fig, output)


def hedge(data, output, phase):
    fig, ax = plt.subplots(figsize=(6.45, 3.65))
    fig.subplots_adjust(left=.185, right=.93, top=.76, bottom=.27)
    heading(fig, "Content 与三个传统基准的对冲能量差", SUBTITLE[phase] + " · 主块长 10",
            "圆点为点估计，横线为名义 95% 区间，空心菱形为族内调整单侧上界。\n"
            "负值表示 Content 能量更低。橙色点线为事前点值门 −5%。")
    group = data.loc["hedge"].loc[FINANCIAL]
    intervals(ax, group, FINANCIAL, -.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("各子期平均能量比减一，再等权平均", labelpad=7)
    save(fig, output)


def losses(run, output, phase):
    trace = pd.read_csv(run / "training_trace.csv", dtype={"fold": str})
    fold_set = FOLDS[phase]
    expected = {(fold, model, seed, epoch) for fold in fold_set for model in SCORERS
                for seed in SEEDS for epoch in range(1, 41)}
    keys = ["fold", "model", "seed", "epoch"]
    measures = ["fixed_train_normalized_matrix_qlike", "validation_normalized_matrix_qlike"]
    if (set(trace.phase) != {phase} or trace.duplicated(keys).any()
        or set(trace[keys].itertuples(index=False, name=None)) != expected
        or not np.isfinite(trace[measures]).all().all()):
        raise ValueError("Training trace differs from the frozen fold/seed/epoch contract")
    averaged = trace.groupby(["model", "epoch"])[measures].mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 3.85))
    fig.subplots_adjust(left=.125, right=.96, top=.70, bottom=.23, wspace=.42)
    fold_note = ("中国探索运行 · 每种评分器 3 折 × 3 个种子等权" if phase == "exploration"
                 else "美国外部确认运行 · 每种评分器 3 个种子等权")
    heading(fig, "三种评分器的固定训练集与验证集损失", fold_note,
            "按相同轮次汇总。两图使用各自纵轴范围，分别对应不同请求集，不能解读为单个模型的泛化差距。")
    lines = []
    for ax, measure, title in zip(axes, measures, ["固定完整训练集", "验证集"]):
        for model, linestyle in zip(SCORERS, ["--", "-.", "-"]):
            series = averaged.loc[model, measure]
            (line,) = ax.plot(series.index, series, color=PALETTE[model], linestyle=linestyle,
                              linewidth=1.45, label=model)
            if ax == axes[0]:
                lines.append(line)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlabel("轮次")
        ax.set_ylabel("归一化矩阵 QLIKE")
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        style_axes(ax, horizontal=False)
    fig.legend(handles=lines, loc="upper center", bbox_to_anchor=(.58, .831), ncol=3,
               fontsize=8.6, frameon=False)
    save(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=FOLDS, default="exploration")
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "confirmation" and not all([args.analysis, args.run, args.output]):
        parser.error("Confirmation requires explicit analysis, run, and output paths")
    analysis = args.analysis or ROOT / "outputs" / "china_analysis_001"
    run = args.run or ROOT / "outputs" / "china_matrix_001"
    output = args.output or FIGURES
    for folder, file_name in [(analysis, "analysis_manifest.json"), (run, "manifest.json")]:
        if json.loads((folder / file_name).read_text(encoding="utf-8"))["phase"] != args.phase:
            raise ValueError(f"{file_name} does not match requested phase")
    setup_style()
    data = read_intervals(analysis)
    output.mkdir(parents=True, exist_ok=True)
    results(analysis, output / "v4_results.png", args.phase)
    pairs(data, output / "v4_pairs.png", args.phase)
    hedge(data, output / "v4_hedge.png", args.phase)
    losses(run, output / "v4_losses.png", args.phase)


if __name__ == "__main__":
    main()
