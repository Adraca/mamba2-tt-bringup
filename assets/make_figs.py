#!/usr/bin/env python3
"""Generate the two showcase figures for the Mamba-2 post: the SSD chunked-scan diagram + the sim results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

INK = "#1b2430"; MUL = "#2e7d5b"; MULF = "#d6efe3"; REC = "#b5651d"; RECF = "#f5e3d0"
GRID = "#c9d1d9"; ACC = "#2f6db3"

# ---------------------------------------------------------------- diagram
def box(ax, x, y, w, h, text, fc, ec, fs=10, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight)

def arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2))

def diagram():
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=150)
    ax.set_xlim(0, 11.5); ax.set_ylim(-0.35, 6.0); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.2, 5.6, "Mamba-2 SSD scan", fontsize=18, weight="bold", color=INK)
    ax.text(0.2, 5.2, "the linear recurrence, rewritten as a chunked, matmul-heavy computation",
            fontsize=11.5, color="#4a5560")
    ax.text(0.2, 4.6, "sequence  →  chunks of Q tokens", fontsize=10.5, color="#4a5560")

    cx = [0.4, 3.15, 5.9]; cw = 2.4          # 3 chunks; end at 2.8 / 5.55 / 8.3
    oy_chunk, oy_diag, sy = 3.75, 2.6, 1.35
    for i, x in enumerate(cx):
        box(ax, x, oy_chunk, cw, 0.55, f"chunk {i}", "#eef2f6", GRID, fs=10.5, weight="bold")
        box(ax, x, oy_diag, cw, 0.85, "diagonal block\n$(C\\,B^{T}\\!\\odot L)\\,x$", MULF, MUL, fs=10)
        arrow(ax, (x+cw/2, oy_chunk-0.02), (x+cw/2, oy_diag+0.87), color=INK)

    # chunk states + inter-chunk recurrence
    for i, x in enumerate(cx):
        box(ax, x+cw/2-0.55, sy, 1.1, 0.62, f"state $S_{i}$", RECF, REC, fs=10)
        arrow(ax, (x+cw/2-0.18, oy_diag-0.02), (x+cw/2-0.18, sy+0.64), color=REC, ls=(0,(4,2)))
        arrow(ax, (x+cw/2+0.18, sy+0.66), (x+cw/2+0.18, oy_diag-0.02), color=MUL, ls=(0,(1,1)))
    for i in range(len(cx)-1):
        arrow(ax, (cx[i]+cw/2+0.55, sy+0.31), (cx[i+1]+cw/2-0.55, sy+0.31), color=REC, lw=2.2)
    ax.text(0.4, sy-0.55, "inter-chunk recurrence — only $C$ chunk-steps, not $L$ tokens (cheap)",
            fontsize=9.5, color=REC)
    ax.text(0.4, sy-0.95, "off-diagonal read  $C\\cdot S$  [matmul] — carried-in state feeds each chunk's output",
            fontsize=9.5, color=MUL)

    # output column (its own space, no overlap)
    ox = 9.1
    box(ax, ox, oy_diag+0.05, 2.0, 0.75, "output $Y$\n$=Y_{diag}+Y_{off}$", "#eef2f6", GRID, fs=10.5, weight="bold")
    arrow(ax, (cx[-1]+cw, oy_diag+0.42), (ox, oy_diag+0.42), color=INK)

    # legend
    box(ax, 0.4, -0.25, 0.3, 0.28, "", MULF, MUL)
    ax.text(0.8, -0.11, "matmul — heavy compute, maps to the Tensix matrix engine", fontsize=9.5, color=INK, va="center")
    box(ax, 7.0, -0.25, 0.3, 0.28, "", RECF, REC)
    ax.text(7.4, -0.11, "short recurrence — cheap", fontsize=9.5, color=INK, va="center")

    fig.savefig("ssd_scan_diagram.png", bbox_inches="tight", facecolor="white")
    print("wrote ssd_scan_diagram.png")

# ---------------------------------------------------------------- results
def results():
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    fig.patch.set_facecolor("white")
    labels = ["M1 · SSD diagonal block", "M2 · full multi-chunk SSD scan", "M3 · full Mamba-2 mixer block"]
    pcc = [1.000000, 1.000000, 0.999988]
    y = np.arange(len(labels))[::-1]
    lo = 0.985
    ax.barh(y, [p-lo for p in pcc], left=lo, height=0.5, color="#2e7d5b", edgecolor="#1f5a41")
    ax.axvline(0.99, color="#b5651d", lw=2, ls="--")
    ax.text(0.99, len(labels)-0.35, " TT bring-up gate  PCC ≥ 0.99", color="#b5651d", fontsize=10, va="bottom")
    for yi, (l, p) in zip(y, zip(labels, pcc)):
        ax.text(lo+0.0005, yi, l, va="center", ha="left", fontsize=10.5, color="white", weight="bold")
        ax.text(1.0002, yi, f"{p:.6f}", va="center", ha="left", fontsize=10.5, color="#1b2430", weight="bold")
    ax.set_xlim(lo, 1.006); ax.set_ylim(-0.6, len(labels)-0.1)
    ax.set_yticks([]); ax.set_xticks([0.985, 0.99, 0.995, 1.0])
    ax.set_xlabel("PCC vs. an independent reference — on the ttsim functional simulator, no hardware", fontsize=10)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("Mamba-2 on Tenstorrent — sim-validated, bit-exact fp32", fontsize=14, weight="bold", loc="left", color="#1b2430")
    fig.savefig("mamba2_sim_results.png", bbox_inches="tight", facecolor="white")
    print("wrote mamba2_sim_results.png")

if __name__ == "__main__":
    diagram(); results()
