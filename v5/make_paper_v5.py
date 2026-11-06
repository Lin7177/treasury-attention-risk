"""Plot the V5 paired contrasts from an authorized aggregate summary."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "v5_run"
FIGURE = ROOT / "figures" / "v5_pairs.png"
LABEL = {"TrainingFixed":"训练固定", "ContentNoQuery":"关闭查询", "MemoryKernel":"案例核",
         "MemoryRecency":"案例时间权重", "MemoryUniform":"案例等权", "Ridge":"岭回归",
         "HAR_CCC":"HAR-CCC", "GARCH_CCC":"GARCH-CCC", "Kernel":"V4固定核",
         "Uniform":"V4均匀", "Prior":"固定先验"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def plot(summary, output, *, synthetic=False):
    result = read(summary)
    data = {(r["market"], r["baseline"]): r for r in result["pairs"] if r["block"] == 10}
    known_fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ["Microsoft YaHei","SimHei","Noto Sans CJK SC"]
                                       if f in known_fonts), "DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False
    models = ["DCC","Kernel","MemoryKernel","MemoryUniform","MLP","ContentNoQuery","Ridge"]
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 3.3), sharey=True)
    for ax, market, title in zip(axes, ("us","china"), ("美国历史探索","中国历史转移")):
        rows = [data[market, x] for x in models]
        point, lo, hi = [100*np.array([r[key] for r in rows]) for key in ("point","lower","upper")]
        y = np.arange(len(models))
        ax.axvline(0,color="#8b95a1",lw=1)
        ax.errorbar(point,y,xerr=[point-lo,hi-point],fmt="o",color="#164768",capsize=2,ms=4,lw=1)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("风险差（%）",fontsize=9)
        ax.grid(axis="x",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_yticks(np.arange(len(models)),[LABEL.get(x,x) for x in models])
    axes[0].invert_yaxis()
    if synthetic:
        fig.suptitle('SYNTHETIC FIXTURE - NOT RESEARCH EVIDENCE')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=270,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
