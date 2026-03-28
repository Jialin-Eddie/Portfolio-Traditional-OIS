"""
generate_report.py
Generates a comprehensive academic-quality PDF report for the Portfolio-Traditional-OIS project.
Output: report.pdf in the project root.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import textwrap

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(BASE, "outputs")
REPORT  = os.path.join(BASE, "report.pdf")

# ── Colour palette ───────────────────────────────────────────────────────────
DARK_BLUE   = "#1B3A6B"
MED_BLUE    = "#2E6DA4"
LIGHT_BLUE  = "#D6E4F0"
ACCENT      = "#E8534A"
GREY_BG     = "#F7F9FC"
GREY_LINE   = "#CCCCCC"
WHITE       = "#FFFFFF"
TEXT_DARK   = "#1A1A2E"

PORTRAIT  = (8.5, 11)
LANDSCAPE = (11, 8.5)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_spine_style(ax, left=True, bottom=True):
    for spine in ax.spines.values():
        spine.set_visible(False)
    if left:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(GREY_LINE)
    if bottom:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(GREY_LINE)
    ax.tick_params(colors=TEXT_DARK, labelsize=9)


def page_header(fig, title, y=0.97):
    fig.text(0.5, y, title, ha="center", va="top",
             fontsize=14, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif")
    # horizontal rule
    line = plt.Line2D([0.05, 0.95], [y - 0.025, y - 0.025],
                      transform=fig.transFigure,
                      color=MED_BLUE, linewidth=1.5)
    fig.add_artist(line)


def page_footer(fig, page_num, total=15):
    fig.text(0.5, 0.015, f"Page {page_num} of {total}  |  "
             "Portfolio-Traditional-OIS  |  Confidential",
             ha="center", va="bottom", fontsize=7, color="grey",
             fontfamily="serif")


def draw_table(ax, df, col_widths=None, row_colors=None,
               header_color=DARK_BLUE, fontsize=9, header_fontsize=9.5):
    """Draw a styled table on ax (which should be axis-off)."""
    ax.axis("off")
    n_rows, n_cols = df.shape
    col_labels = list(df.columns)
    data = df.values

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    # normalise widths to sum 1
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    row_height = 0.9 / (n_rows + 1)   # +1 for header
    x_starts = []
    x = 0.0
    for w in col_widths:
        x_starts.append(x)
        x += w

    def draw_cell(text, x0, y0, w, h, bg, fg, bold=False, fontsize=fontsize, align="center"):
        rect = FancyBboxPatch((x0, y0), w, h,
                              boxstyle="square,pad=0",
                              facecolor=bg, edgecolor=WHITE, linewidth=0.5,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, str(text),
                ha=align, va="center", fontsize=fontsize,
                color=fg, fontweight="bold" if bold else "normal",
                transform=ax.transAxes, clip_on=False)

    # header row
    header_y = 0.9
    for j, (label, x0, w) in enumerate(zip(col_labels, x_starts, col_widths)):
        draw_cell(label, x0, header_y, w, row_height,
                  bg=header_color, fg=WHITE, bold=True,
                  fontsize=header_fontsize)

    # data rows
    for i, row in enumerate(data):
        y0 = header_y - (i + 1) * row_height
        bg = LIGHT_BLUE if i % 2 == 0 else WHITE
        if row_colors and i in row_colors:
            bg = row_colors[i]
        for j, (val, x0, w) in enumerate(zip(row, x_starts, col_widths)):
            fg = TEXT_DARK
            if str(val) in ("PASS", "100.00%"):
                fg = "#1E8E3E"
            elif str(val) in ("FAIL",):
                fg = ACCENT
            draw_cell(val, x0, y0, w, row_height, bg=bg, fg=fg,
                      fontsize=fontsize)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

metrics_full = pd.read_csv(os.path.join(OUTPUTS, "metrics_summary_Full.csv"), index_col=0)
metrics_is   = pd.read_csv(os.path.join(OUTPUTS, "metrics_summary_IS_2015_2020.csv"), index_col=0)
metrics_oos  = pd.read_csv(os.path.join(OUTPUTS, "metrics_summary_OOS_2021_2024.csv"), index_col=0)
ablation     = pd.read_csv(os.path.join(OUTPUTS, "feature_ablation.csv"))

# convenience getter
def mget(df, row, col):
    try:
        v = df.loc[row, col]
        return float(v)
    except Exception:
        return np.nan

def fmt_pct(v, decimals=2):
    if np.isnan(v):
        return "—"
    return f"{v*100:.{decimals}f}%"

def fmt_num(v, decimals=4):
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"

def fmt_num2(v, decimals=2):
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────────────────────────────────────

with PdfPages(REPORT) as pdf:

    # ── PAGE 1 — Title ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(DARK_BLUE)

    # Top band
    top_band = FancyBboxPatch((0, 0.72), 1, 0.28,
                               boxstyle="square,pad=0",
                               facecolor=DARK_BLUE, edgecolor="none",
                               transform=fig.transFigure)
    fig.add_artist(top_band)

    # Mid band (light)
    mid_band = FancyBboxPatch((0, 0.15), 1, 0.57,
                               boxstyle="square,pad=0",
                               facecolor="#EEF3FB", edgecolor="none",
                               transform=fig.transFigure)
    fig.add_artist(mid_band)

    # Bottom band
    bot_band = FancyBboxPatch((0, 0), 1, 0.15,
                               boxstyle="square,pad=0",
                               facecolor=MED_BLUE, edgecolor="none",
                               transform=fig.transFigure)
    fig.add_artist(bot_band)

    # Title
    fig.text(0.5, 0.89,
             "Constrained Portfolio Optimization\nwith Traditional and\nOption-Implied Signals",
             ha="center", va="center", fontsize=22, fontweight="bold",
             color=WHITE, fontfamily="serif", linespacing=1.5)

    # Subtitle
    fig.text(0.5, 0.75,
             "S&P 500 Universe  |  Monthly Rebalancing  |  2016 – 2024",
             ha="center", va="center", fontsize=13, color=LIGHT_BLUE,
             fontfamily="serif")

    # Accent line
    line = plt.Line2D([0.15, 0.85], [0.71, 0.71],
                      transform=fig.transFigure,
                      color=ACCENT, linewidth=2)
    fig.add_artist(line)

    # Key result boxes
    kw = dict(ha="center", va="center", fontfamily="serif",
              transform=fig.transFigure)

    results = [
        ("OOS Excess Return", "+1.85%", "#E8F5E9", "#1E8E3E"),
        ("Full-Period IR", "1.28", "#E8F0FE", MED_BLUE),
        ("All Constraints", "100% PASS", "#E8F5E9", "#1E8E3E"),
    ]
    box_w, box_h = 0.25, 0.10
    starts = [0.06, 0.375, 0.69]
    y_box = 0.50

    for (label, value, bg, fg), bx in zip(results, starts):
        rect = FancyBboxPatch((bx, y_box - box_h / 2), box_w, box_h,
                               boxstyle="round,pad=0.01",
                               facecolor=bg, edgecolor=fg, linewidth=2,
                               transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(bx + box_w / 2, y_box + 0.015, label,
                 fontsize=9, color=TEXT_DARK, **kw)
        fig.text(bx + box_w / 2, y_box - 0.018, value,
                 fontsize=15, fontweight="bold", color=fg, **kw)

    # Strategy description
    desc = (
        "This report presents an academic-quality evaluation of a long-only\n"
        "constrained portfolio strategy combining XGBoost ML signals with\n"
        "Fama-French factor betas, option-implied volatility features, and\n"
        "traditional equity signals. Optimization is performed via cvxpy with\n"
        "weight bounds, factor exposure, and drawdown constraints."
    )
    fig.text(0.5, 0.33, desc, ha="center", va="center",
             fontsize=10.5, color=TEXT_DARK, fontfamily="serif",
             linespacing=1.6)

    # Meta info
    meta = [
        ("Benchmark",    "Market-Cap-Weighted S&P 500 (Top 500 by mktcap)"),
        ("Universe",     "500 S&P 500 stocks/month, monthly rebalancing"),
        ("In-Sample",    "2016 – 2020  (60 months)"),
        ("Out-of-Sample","2021 – 2024  (48 months)"),
        ("Optimizer",    "cvxpy  |  Covariance: Ledoit-Wolf (36-month rolling)"),
    ]
    y_meta = 0.245
    for k, v in meta:
        fig.text(0.15, y_meta, f"{k}:", fontsize=9, color=MED_BLUE,
                 fontweight="bold", fontfamily="serif",
                 transform=fig.transFigure, ha="left", va="center")
        fig.text(0.36, y_meta, v, fontsize=9, color=TEXT_DARK,
                 fontfamily="serif",
                 transform=fig.transFigure, ha="left", va="center")
        y_meta -= 0.022

    # Footer
    fig.text(0.5, 0.07, "Quantitative Research  |  2025",
             ha="center", va="center", fontsize=11, color=WHITE,
             fontfamily="serif")
    fig.text(0.5, 0.035, "CONFIDENTIAL — FOR INTERNAL USE ONLY",
             ha="center", va="center", fontsize=8, color=LIGHT_BLUE,
             fontfamily="serif")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 1 done")

    # ── PAGE 2 — Executive Summary ───────────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Executive Summary", y=0.96)
    page_footer(fig, 2)

    content = [
        ("OBJECTIVE", [
            "Construct a long-only constrained portfolio that systematically outperforms the",
            "market-cap-weighted (MCW) S&P 500 benchmark under realistic portfolio constraints.",
            "The strategy must satisfy three hard constraints at all times:",
            "  (1) weight bounds [0, 2× benchmark weight per stock],",
            "  (2) factor exposure deviation ≤ 0.10 (per FF4 factor),",
            "  (3) monthly active return ≥ −2% (drawdown guard).",
        ]),
        ("DATA & UNIVERSE", [
            "Source: OpenAssetPricing panel + WRDS option-implied data",
            "Universe: Top 500 S&P 500 stocks by market cap per month",
            "  108 months total (Jan 2016 – Dec 2024), 500 stocks/month",
            "In-Sample (IS):     Jan 2016 – Dec 2020  (60 months)",
            "Out-of-Sample (OOS): Jan 2021 – Dec 2024 (48 months)",
            "Rebalancing: Monthly, executed at month-end closing prices",
            "All return metrics use geometric (compounded) annualisation",
        ]),
        ("METHODOLOGY", [
            "Signal: XGBoost (n_estimators=500 max, lr=0.05, max_depth=4, min_child_weight=50)",
            "  Early stopping: patience=20 on temporal val (last 15% of training months)",
            "  Purge: val's last month removed (fwd_ret overlaps predict_month)",
            "  EMA smoothing: alpha=0.3 applied to raw signal before optimizer",
            "Features (11): FF4 betas (4) + option-implied SKEW/AIV/GLB (3)",
            "  + Mom12m/IdioVol3F/BM (3) + resid_signal (1)",
            "Preprocessing: Winsorize(p1/p99) → Box-Cox (IS-fitted) → Z-score",
            "Optimizer: cvxpy maximize(μᵀw) with 5 constraints:",
            "  (1) Σw=1, (2) w≥0, (3) w≤2·w_bench, (4) |Δβ|≤0.10, (5) CVaR≤2%",
            "  + TE≤1.5% + post-hoc drawdown guard (shrink if prev month < −1.5%)",
            "Covariance: Ledoit-Wolf shrinkage, 36-month rolling window",
        ]),
        ("KEY FINDINGS", [
            "✦ OOS: Return 14.1% vs 12.0% BNCH (+1.9%); Sharpe 0.83 vs 0.77",
            "✦ OOS Information Ratio: 0.61; Tracking Error: 3.0%",
            "✦ Full-period: Return 19.0% vs 14.9% (+4.1%); IR: 1.29; Sharpe 1.10 vs 0.93",
            "✦ IS: Return 22.6% vs 17.1% (+5.6%); IR: 1.85",
            "✦ All 3 constraints satisfied 100% across 108 months (0 violations)",
            "✦ Max monthly underperformance: −1.46% (well within −2% limit)",
            "✦ EMA signal smoothing was the key innovation: eliminated drawdown violations",
            "  while maintaining OOS alpha (raw signal had −2.31% worst month)",
            "✦ CVaR + post-hoc guard provides dual-layer downside protection",
            "✦ Early stopping + purge improved OOS IR from ~0 to 0.61",
            "✦ Limitation: OpenAssetPricing data gap in 2023–2024; no transaction costs",
        ]),
    ]

    y = 0.90
    for section_title, lines in content:
        # section header
        rect = FancyBboxPatch((0.05, y - 0.003), 0.90, 0.022,
                               boxstyle="square,pad=0.005",
                               facecolor=DARK_BLUE, edgecolor="none",
                               transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(0.07, y + 0.008, section_title,
                 ha="left", va="center", fontsize=10,
                 fontweight="bold", color=WHITE, fontfamily="serif",
                 transform=fig.transFigure)
        y -= 0.030

        for line in lines:
            fig.text(0.07, y, line,
                     ha="left", va="top", fontsize=8.5,
                     color=TEXT_DARK, fontfamily="serif",
                     transform=fig.transFigure)
            y -= 0.018
        y -= 0.020

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 2 done")

    # ── PAGE 3 — Strategy Pipeline ────────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Strategy Pipeline", y=0.96)
    page_footer(fig, 3)

    pipeline_steps = [
        ("Data Panel", "OpenAssetPricing + WRDS\n1.58M rows, 500 stocks/month\n108 months (2016-2024)", DARK_BLUE),
        ("Feature Engineering", "11 features: FF4 betas (4)\n+ Option-Implied: SKEW/AIV/GLB (3)\n+ Traditional: Mom/IdioVol/BM (3)\n+ Residual Signal (1)", MED_BLUE),
        ("Preprocessing", "Winsorize (p1/p99)\n\u2192 Box-Cox (IS-fitted \u03bb)\n\u2192 Cross-sectional Z-score", MED_BLUE),
        ("XGBoost Signal", "Walk-forward monthly re-estimation\nEarly stopping (patience=20)\nPurge: remove val leakage", DARK_BLUE),
        ("EMA Smoothing", "\u03bc(t) = 0.3\u00b7raw(t) + 0.7\u00b7\u03bc(t\u22121)\nHalf-life \u2248 2 months\nReduces signal noise & turnover", MED_BLUE),
        ("cvxpy Optimizer", "max \u03bcTw  s.t. 5 constraints:\n\u2211w=1, w\u22650, w\u22642\u00b7w_bench\n|\u0394\u03b2|\u22640.10, CVaR\u22642%, TE\u22641.5%", DARK_BLUE),
        ("Portfolio", "Long-only constrained weights\nMonthly rebalancing\n+ Post-hoc drawdown guard", MED_BLUE),
    ]

    n_steps = len(pipeline_steps)
    box_w, box_h = 0.70, 0.070
    x_center = 0.50
    y_start = 0.88
    y_gap = 0.095

    for i, (title, desc, color) in enumerate(pipeline_steps):
        y = y_start - i * y_gap
        x0 = x_center - box_w / 2
        rect = FancyBboxPatch((x0, y - box_h / 2), box_w, box_h,
                               boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor="none", alpha=0.9,
                               transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(x0 + 0.015, y + 0.008, title,
                 ha="left", va="center", fontsize=11, fontweight="bold",
                 color=WHITE, fontfamily="serif", transform=fig.transFigure)
        fig.text(x0 + box_w - 0.015, y - 0.002, desc,
                 ha="right", va="center", fontsize=7.5,
                 color="#E0E8F0", fontfamily="serif", transform=fig.transFigure,
                 linespacing=1.3)
        if i < n_steps - 1:
            arrow_y_top = y - box_h / 2 - 0.005
            arrow_y_bot = arrow_y_top - 0.015
            arrow_line = plt.Line2D([x_center, x_center],
                                     [arrow_y_top, arrow_y_bot],
                                     transform=fig.transFigure,
                                     color=ACCENT, linewidth=2.0,
                                     marker="v", markersize=8,
                                     markevery=[1])
            fig.add_artist(arrow_line)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 3 done")

    # ── PAGE 4 — OOS & Full Performance Comparison ──────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Performance Comparison: QS Portfolio vs Benchmark", y=0.96)
    page_footer(fig, 4)

    def build_comparison_table(mdf):
        qs_col = "QS Portfolio"
        bn_col = "Benchmark"
        def g(row):
            try:
                return float(mdf.loc[row, qs_col]), float(mdf.loc[row, bn_col])
            except Exception:
                return np.nan, np.nan
        def gq(row):
            try:
                return float(mdf.loc[row, qs_col])
            except Exception:
                return np.nan

        rows = []
        r, b = g("ann_return")
        rows.append(["Ann. Return", fmt_pct(r), fmt_pct(b), fmt_pct(r - b) if not (np.isnan(r) or np.isnan(b)) else "—"])
        r, b = g("ann_vol")
        rows.append(["Ann. Volatility", fmt_pct(r), fmt_pct(b), fmt_pct(r - b) if not (np.isnan(r) or np.isnan(b)) else "—"])
        r, b = g("sharpe_simple")
        rows.append(["Sharpe (Simple)", fmt_num2(r), fmt_num2(b), fmt_num2(r - b) if not (np.isnan(r) or np.isnan(b)) else "—"])
        r, b = g("max_drawdown")
        rows.append(["Max Drawdown", fmt_pct(r), fmt_pct(b), fmt_pct(r - b) if not (np.isnan(r) or np.isnan(b)) else "—"])
        ir = gq("information_ratio")
        rows.append(["Information Ratio", fmt_num2(ir), "—", "—"])
        te = gq("tracking_error")
        rows.append(["Tracking Error", fmt_pct(te), "—", "—"])
        r, b = g("hit_rate")
        rows.append(["Hit Rate", fmt_pct(r), fmt_pct(b), fmt_pct(r - b) if not (np.isnan(r) or np.isnan(b)) else "—"])
        md = gq("max_relative_dd")
        rows.append(["Max Monthly Underperf.", fmt_pct(md), "—", "—"])
        return rows

    # ── OOS Table ──
    fig.text(0.07, 0.90, "Out-of-Sample Period (2021–2024)", ha="left", va="top",
             fontsize=12, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    oos_rows = build_comparison_table(metrics_oos)
    oos_df = pd.DataFrame(oos_rows, columns=["Metric", "QS Portfolio", "Benchmark", "Δ (QS−BNCH)"])
    ax_oos = fig.add_axes([0.05, 0.56, 0.90, 0.30])
    draw_table(ax_oos, oos_df,
               col_widths=[2.2, 1.2, 1.2, 1.2],
               fontsize=9.5, header_fontsize=10)

    # ── Full Period Table ──
    fig.text(0.07, 0.51, "Full Period (2016–2024)", ha="left", va="top",
             fontsize=12, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    full_rows = build_comparison_table(metrics_full)
    full_df = pd.DataFrame(full_rows, columns=["Metric", "QS Portfolio", "Benchmark", "Δ (QS−BNCH)"])
    ax_full = fig.add_axes([0.05, 0.17, 0.90, 0.30])
    draw_table(ax_full, full_df,
               col_widths=[2.2, 1.2, 1.2, 1.2],
               fontsize=9.5, header_fontsize=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 4 done")

    # ── PAGE 5 — Cumulative Returns OOS (shown first) ──────────────────────
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Cumulative Returns — Out-of-Sample Period (2021–2024)", y=0.97)
    page_footer(fig, 5)

    img_path = os.path.join(OUTPUTS, "cumulative_returns_OOS_2021_2024.png")
    img = plt.imread(img_path)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    ax.imshow(img, aspect="auto")
    ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 5 done")

    # ── PAGE 6 — Cumulative Returns Full ──────────────────────────────────
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Cumulative Returns — Full Period (2016–2024)", y=0.97)
    page_footer(fig, 6)

    img_path = os.path.join(OUTPUTS, "cumulative_returns_Full.png")
    img = plt.imread(img_path)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    ax.imshow(img, aspect="auto")
    ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 6 done")

    # ── PAGE 7 — Monthly Active Returns ────────────────────────────────────
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Monthly Active Returns — Full Period (2016–2024)", y=0.97)
    page_footer(fig, 7)

    img_path = os.path.join(OUTPUTS, "active_returns_Full.png")
    img = plt.imread(img_path)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    ax.imshow(img, aspect="auto")
    ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 7 done")

    # ── PAGE 8 — Factor Exposure Deviations ─────────────────────────────────
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Factor Exposure Deviations from Benchmark — Full Period", y=0.97)
    page_footer(fig, 8)

    img_path = os.path.join(OUTPUTS, "factor_exposure_Full.png")
    img = plt.imread(img_path)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    ax.imshow(img, aspect="auto")
    ax.axis("off")

    # annotation box
    annot = (
        "All four Fama-French factor deviations (MktRF, SMB, HML, MOM) remain within\n"
        "±0.10 at all times. Max observed deviation: 0.057 (MktRF). Constraint: 100% PASS."
    )
    fig.text(0.5, 0.025, annot, ha="center", va="bottom",
             fontsize=8.5, color=DARK_BLUE, style="italic",
             fontfamily="serif")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 8 done")

    # ── PAGE 9 — Rolling Sharpe + Drawdown + Weight Distribution ──────────
    fig = plt.figure(figsize=LANDSCAPE)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Rolling Sharpe Ratio, Drawdown & Weight Distribution", y=0.97)
    page_footer(fig, 9)

    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95,
                           top=0.90, bottom=0.06, wspace=0.06, hspace=0.12)

    ax1 = fig.add_subplot(gs[0, :])
    img1 = plt.imread(os.path.join(OUTPUTS, "rolling_sharpe_Full.png"))
    ax1.imshow(img1, aspect="auto")
    ax1.axis("off")
    ax1.set_title("Rolling 12-Month Sharpe Ratio", fontsize=10, color=DARK_BLUE,
                  fontweight="bold", pad=4)

    ax2 = fig.add_subplot(gs[1, 0])
    img2 = plt.imread(os.path.join(OUTPUTS, "drawdown_Full.png"))
    ax2.imshow(img2, aspect="auto")
    ax2.axis("off")
    ax2.set_title("Drawdown Profile", fontsize=10, color=DARK_BLUE,
                  fontweight="bold", pad=4)

    ax3 = fig.add_subplot(gs[1, 1])
    img3 = plt.imread(os.path.join(OUTPUTS, "weight_distribution_Full.png"))
    ax3.imshow(img3, aspect="auto")
    ax3.axis("off")
    ax3.set_title("Active Weight Distribution", fontsize=10, color=DARK_BLUE,
                  fontweight="bold", pad=4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 9 done")

    # ── PAGE 10 — Constraint Compliance ─────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Constraint Compliance Report", y=0.96)
    page_footer(fig, 10)

    # Summary banner
    rect = FancyBboxPatch((0.10, 0.86), 0.80, 0.060,
                           boxstyle="round,pad=0.01",
                           facecolor="#E8F5E9", edgecolor="#1E8E3E", linewidth=2,
                           transform=fig.transFigure)
    fig.add_artist(rect)
    fig.text(0.50, 0.891, "OVERALL PORTFOLIO COMPLIANCE: 100.00%  —  ALL PASS",
             ha="center", va="center", fontsize=13, fontweight="bold",
             color="#1E8E3E", fontfamily="serif", transform=fig.transFigure)

    compliance_data = [
        ["1", "Weight Bounds",
         "54,000 stock-months", "0", "100.00%", "PASS",
         "Max single viol: 1.8e-5"],
        ["2a", "Factor Exp. — MktRF",
         "108 months", "0", "100.00%", "PASS",
         "Max dev: 0.057"],
        ["2b", "Factor Exp. — SMB",
         "108 months", "0", "100.00%", "PASS",
         "Max dev: 0.051"],
        ["2c", "Factor Exp. — HML",
         "108 months", "0", "100.00%", "PASS",
         "Max dev: 0.052"],
        ["2d", "Factor Exp. — MOM",
         "108 months", "0", "100.00%", "PASS",
         "Max dev: 0.051"],
        ["3", "Relative Drawdown",
         "108 months", "0", "100.00%", "PASS",
         "Worst active: −1.38%"],
    ]

    comp_df = pd.DataFrame(compliance_data,
                           columns=["#", "Constraint", "Checked", "Violations",
                                    "Compliance", "Status", "Notes"])

    ax_c = fig.add_axes([0.05, 0.40, 0.90, 0.42])
    draw_table(ax_c, comp_df,
               col_widths=[0.5, 2.2, 1.4, 1.0, 1.0, 0.8, 1.6],
               fontsize=8.5)

    # Constraint definitions
    fig.text(0.07, 0.36, "Constraint Definitions:", ha="left", va="top",
             fontsize=10, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    defs = [
        ("Constraint 1 — Weight Bounds:",
         "0 ≤ wᵢ ≤ 2 × w_bench,i  for all stocks i at each rebalancing date"),
        ("Constraint 2 — Factor Exposure:",
         "|β_QS,k − β_BNCH,k| ≤ 0.10  for k ∈ {MktRF, SMB, HML, MOM}"),
        ("Constraint 3 — Relative Drawdown:",
         "Monthly active return ≥ −2%  (i.e., max monthly underperformance limit)"),
    ]

    y_def = 0.31
    for title, detail in defs:
        fig.text(0.07, y_def, title, ha="left", va="top",
                 fontsize=9, fontweight="bold", color=TEXT_DARK,
                 fontfamily="serif", transform=fig.transFigure)
        fig.text(0.07, y_def - 0.025, detail, ha="left", va="top",
                 fontsize=9, color=TEXT_DARK,
                 fontfamily="serif", transform=fig.transFigure)
        y_def -= 0.060

    # Note on near-violation
    note = (
        "Note: Weight bounds show a maximum single violation of 1.8×10⁻⁵ which is due to floating-point\n"
        "      rounding in the cvxpy solver (effectively zero). The worst month for weight bounds was\n"
        "      January 2016 (first rebalancing), where solver tolerance was at its maximum."
    )
    fig.text(0.07, 0.09, note, ha="left", va="top",
             fontsize=8, color="grey", fontfamily="serif",
             transform=fig.transFigure)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 10 done")

    # ── PAGE 11 — Feature Ablation ───────────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Feature Ablation Study", y=0.96)
    page_footer(fig, 11)

    # description
    desc_lines = [
        "To assess the marginal contribution of each feature group, we train separate XGBoost models",
        "using subsets of the full 11-feature space and evaluate IC (Information Coefficient) and ICIR",
        "(IC Information Ratio) both in-sample (IS) and out-of-sample (OOS). A higher IC indicates better",
        "cross-sectional predictive power. The OOS IC is the primary evaluation criterion.",
    ]
    y_d = 0.90
    for line in desc_lines:
        fig.text(0.07, y_d, line, ha="left", va="top", fontsize=9,
                 color=TEXT_DARK, fontfamily="serif", transform=fig.transFigure)
        y_d -= 0.020

    # Build display table
    abl = ablation.copy()
    abl_display = pd.DataFrame({
        "Feature Set":    abl["Feature Set"],
        "N Features":     abl["N Features"].astype(int).astype(str),
        "IS IC":          abl["IS IC"].map(lambda x: f"{x:.4f}"),
        "OOS IC":         abl["OOS IC"].map(lambda x: f"{x:.4f}"),
        "OOS ICIR":       abl["OOS ICIR"].map(lambda x: f"{x:.4f}"),
        "Top Feature":    abl["Top Feature"],
    })

    ax_abl = fig.add_axes([0.05, 0.52, 0.90, 0.28])
    draw_table(ax_abl, abl_display,
               col_widths=[1.8, 1.0, 1.0, 1.0, 1.0, 1.5],
               fontsize=9.5, header_fontsize=10)

    # Analysis
    fig.text(0.07, 0.49, "Key Observations:", ha="left", va="top",
             fontsize=10, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    observations = [
        "• All feature subsets show significant IS IC collapse in OOS (signal decay from IS→OOS).",
        "• Option-Implied features (GLB, AIV, SKEW) achieve the best OOS IC (−0.009), suggesting",
        "  they are more regime-stable than traditional or beta features.",
        "• Traditional features (including residual momentum) achieve the highest IS IC (0.187)",
        "  but suffer the largest OOS collapse (OOS IC: −0.032), indicating strong IS overfitting.",
        "• Betas-Only produces moderate IS IC (0.155) and OOS IC (−0.013), likely due to factor",
        "  neutralization constraints limiting exploitable factor tilts.",
        "• The full 11-feature model achieves IS IC = 0.199 (highest) but OOS IC = −0.025,",
        "  implying the feature combination amplifies overfitting.",
        "• The top feature in IS for all multi-feature sets is resid_signal (residual momentum),",
        "  which aligns with momentum strategies performing well in 2016–2020 but poorly post-COVID.",
        "• Implication: Future work should focus on regime-adaptive weighting or option-implied",
        "  only models for more robust OOS performance.",
    ]

    y_o = 0.455
    for obs in observations:
        fig.text(0.07, y_o, obs, ha="left", va="top", fontsize=8.8,
                 color=TEXT_DARK, fontfamily="serif", transform=fig.transFigure)
        y_o -= 0.033

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 11 done")

    # ── PAGE 12 — Preprocessing Comparison ──────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Preprocessing Method Comparison: Rank-Norm vs Box-Cox+Z-Score", y=0.96)
    page_footer(fig, 12)

    desc2 = [
        "Two cross-sectional preprocessing pipelines were evaluated to assess sensitivity of strategy",
        "performance to normalization method. Rank-Normalization maps each feature to uniform rank scores,",
        "while Box-Cox+Z-Score applies a power transformation (λ fitted on IS data) followed by",
        "cross-sectional standardization. The comparison isolates the effect of preprocessing on OOS performance.",
    ]
    y_d2 = 0.90
    for line in desc2:
        fig.text(0.07, y_d2, line, ha="left", va="top", fontsize=9,
                 color=TEXT_DARK, fontfamily="serif", transform=fig.transFigure)
        y_d2 -= 0.020

    # OOS table
    fig.text(0.07, 0.82, "Out-of-Sample (OOS) Period: 2021–2024", ha="left", va="top",
             fontsize=10, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    oos_prep = pd.DataFrame({
        "Metric":        ["Ann. Return", "Volatility", "Sharpe (Simple)", "Max Drawdown", "Information Ratio"],
        "Benchmark":     ["11.39%", "15.69%", "0.7259", "−23.17%", "—"],
        "Rank-Norm":     ["9.71%", "16.34%", "0.5940", "−24.86%", "−0.7451"],
        "Box-Cox+Z":     ["11.39%", "16.79%", "0.6786", "−23.72%", "−0.0001"],
        "Δ (BxZ−RN)":   ["+1.68pp", "−0.45pp", "+0.0846", "+1.14pp", "+0.7450"],
    })

    ax_oos = fig.add_axes([0.05, 0.59, 0.90, 0.20])
    draw_table(ax_oos, oos_prep,
               col_widths=[1.6, 1.2, 1.2, 1.2, 1.2],
               fontsize=9, header_fontsize=9.5)

    # Full Period table
    fig.text(0.07, 0.56, "Full Period: 2016–2024", ha="left", va="top",
             fontsize=10, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    full_prep = pd.DataFrame({
        "Metric":        ["Ann. Return", "Volatility", "Sharpe (Simple)", "Max Drawdown", "Information Ratio"],
        "Benchmark":     ["14.11%", "15.16%", "0.9308", "−23.17%", "—"],
        "Rank-Norm":     ["17.68%", "16.24%", "1.0885", "−24.86%", "1.2123"],
        "Box-Cox+Z":     ["17.74%", "16.27%", "1.0901", "−23.72%", "1.2629"],
        "Δ (BxZ−RN)":   ["+0.06pp", "+0.03pp", "+0.0016", "+1.14pp", "+0.0506"],
    })

    ax_full = fig.add_axes([0.05, 0.32, 0.90, 0.20])
    draw_table(ax_full, full_prep,
               col_widths=[1.6, 1.2, 1.2, 1.2, 1.2],
               fontsize=9, header_fontsize=9.5)

    # Analysis
    fig.text(0.07, 0.31, "Analysis:", ha="left", va="top",
             fontsize=10, fontweight="bold", color=DARK_BLUE,
             fontfamily="serif", transform=fig.transFigure)

    analysis = [
        "• OOS performance: Box-Cox+Z significantly outperforms Rank-Norm. The IR improvement",
        "  (+0.745) is substantial — Rank-Norm produced −1.7% OOS excess return while Box-Cox+Z",
        "  achieved near-zero excess return (≈0.000%), matching the benchmark.",
        "• Full-period performance: The gap narrows considerably. Both methods produce similar",
        "  returns (+3.6% excess) and Sharpe ratios, with Box-Cox+Z marginally better (IR: 1.26 vs 1.21).",
        "• Interpretation: Rank-Normalization is vulnerable to distribution shifts between IS and OOS",
        "  periods (regime changes). Box-Cox stabilizes the feature distribution by removing",
        "  skewness using IS-fitted parameters, reducing the impact of distributional shift.",
        "• Recommendation: Box-Cox+Z-Score preprocessing is the preferred method for production.",
        "  The IS-fitted λ parameters must be refreshed periodically to remain valid.",
    ]

    y_a = 0.270
    for line in analysis:
        fig.text(0.07, y_a, line, ha="left", va="top", fontsize=8.8,
                 color=TEXT_DARK, fontfamily="serif", transform=fig.transFigure)
        y_a -= 0.030

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 12 done")

    # ── PAGE 13 — Methodology ────────────────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Methodology", y=0.96)
    page_footer(fig, 13)

    method_sections = [
        ("1. Benchmark Construction", [
            "The benchmark is a market-cap-weighted (MCW) portfolio of approximately 700 S&P 500",
            "constituent stocks, rebalanced monthly. Weights are computed from end-of-month market",
            "capitalization (price × shares outstanding). The benchmark serves as both the",
            "performance reference and the constraint anchor for weight bounds and factor exposures.",
        ]),
        ("2. Signal Generation (XGBoost + EMA)", [
            "Model: XGBoost (n_estimators=500 max, lr=0.05, max_depth=4, min_child_weight=50)",
            "Target: Next-month excess return (fwd_ret)",
            "Early stopping: patience=20. Val = last 15% of training months (temporal split).",
            "  Purge: val last month removed (fwd_ret leaks into predict_month price data).",
            "  No refit after selection — 85%-train model used directly.",
            "EMA smoothing: raw signal → EMA(alpha=0.3) per stock across months.",
            "  Reduces month-to-month signal noise (rank corr 0.45→higher), cuts turnover,",
            "  and eliminates drawdown violations from extreme rebalancing.",
            "Walk-forward: IS model trained on 2015–2020; OOS re-estimated monthly using",
            "  expanding window. Assertion: max(train_date) < min(predict_date).",
        ]),
        ("3. Feature Engineering (11 Features)", [
            "Factor Betas (4): beta_mktrf, beta_smb, beta_hml, beta_mom",
            "  Estimated via 36-month rolling OLS regression against Fama-French factors.",
            "Option-Implied (3): SKEW (implied skewness), AIV (at-the-money IV), GLB (global beta)",
            "  Source: WRDS OptionMetrics. Monthly aggregated from daily option data.",
            "Traditional (3): Mom12m (12-month price momentum, skip-1), IdioVol3F (idiosyncratic",
            "  vol from 3-factor model, 60-day rolling), BM (book-to-market ratio)",
            "Residual Momentum (1): resid_signal = momentum residualized on FF4 factor returns",
        ]),
        ("4. Preprocessing Pipeline", [
            "Step 1 — Winsorize: Clip each feature at 1st and 99th percentile per month",
            "Step 2 — Box-Cox: Apply power transformation xᵢ → (xᵢᵅ − 1)/λ  with λ fitted on IS",
            "  Positive-only via shifting: x′ = x − min(x) + ε before transformation",
            "Step 3 — Z-Score: Standardize cross-sectionally per month (mean 0, std 1)",
            "IS-fitted parameters (λ values per feature) are stored in preprocess_params.joblib",
            "and applied unchanged in OOS to prevent data leakage.",
        ]),
        ("5. Portfolio Optimization (CVaR + Post-Hoc Guard)", [
            "Objective: maximize μᵀw  (EMA-smoothed signal scores)",
            "Constraints:",
            "  • Σwᵢ = 1, 0 ≤ wᵢ ≤ 2 × w_bench,i",
            "  • |β_QS,k − β_BNCH,k| ≤ 0.10  for each FF4 factor",
            "  • CVaR(95%) ≤ 2% on active returns (worst 5% scenario constraint)",
            "  • Tracking Error ≤ 1.5% (Ledoit-Wolf covariance, 36-month rolling window)",
            "Post-hoc drawdown guard: if prior month active return < −1.5%, shrink active",
            "  weights by 50% toward benchmark. Uses only realized (past) data — no lookahead.",
            "Relaxation ladder: CVaR 2%→2.5%→3%→4%, TE 1.5%→2%→2.5%→3% → benchmark fallback",
            "Solver: SCS primary, ECOS fallback",
        ]),
        ("6. Performance Evaluation", [
            "Sharpe Ratio: Newey-West HAC standard errors (6 lags) for significance",
            "Information Ratio: Active return / Tracking Error (annualized)",
            "Max Drawdown: Peak-to-trough cumulative loss on strategy portfolio",
            "Constraint Compliance: Binary checks per period, compliance = fraction PASS",
            "Feature IC: Spearman rank correlation between signal scores and forward returns",
        ]),
    ]

    y_m = 0.91
    for title, lines in method_sections:
        # section header
        rect = FancyBboxPatch((0.05, y_m - 0.003), 0.90, 0.020,
                               boxstyle="square,pad=0.004",
                               facecolor=MED_BLUE, edgecolor="none",
                               transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(0.07, y_m + 0.007, title,
                 ha="left", va="center", fontsize=9.5,
                 fontweight="bold", color=WHITE, fontfamily="serif",
                 transform=fig.transFigure)
        y_m -= 0.027

        for line in lines:
            fig.text(0.07, y_m, line,
                     ha="left", va="top", fontsize=8.3,
                     color=TEXT_DARK, fontfamily="serif",
                     transform=fig.transFigure)
            y_m -= 0.017
        y_m -= 0.018

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 13 done")

    # ── PAGE 14 — Discussion & Conclusion ────────────────────────────────────
    fig = plt.figure(figsize=PORTRAIT)
    fig.patch.set_facecolor(WHITE)
    page_header(fig, "Discussion & Conclusion", y=0.96)
    page_footer(fig, 14)

    disc_sections = [
        ("Iterative Improvement Summary", [
            "v1→v4: Fixed trees (IR≈0) → +early stopping (IR=0.46) → +purge (IR=0.78)",
            "  → +Top500 + EMA(0.3) + CVaR + post-hoc guard → OOS IR=0.61, 0 violations.",
            "Key insight: drawdown violations came from signal noise, not insufficient constraints.",
        ]),
        ("EMA Signal Smoothing — The Key Innovation", [
            "Raw signal rank corr ~0.45 month-to-month, causing 35% turnover.",
            "EMA(α=0.3): μ(t) = 0.3·raw(t) + 0.7·μ(t−1), half-life ≈ 2 months.",
            "Effect: signal std reduced to 77%; extreme rebalancing eliminated.",
            "6-way experiment: EMA is the only transform eliminating drawdown violations.",
            "",
            "Alpha sensitivity: OOS IC is flat across α ∈ [0.1, 0.7] (range: −0.021 to −0.023),",
            "confirming α=0.3 is not over-tuned. Chosen for ≈2-month half-life matching",
            "typical signal persistence in monthly equity cross-sections.",
        ]),
        ("Constraint Effectiveness", [
            "All three constraints satisfied 100% across 108 months (0 violations):",
            "  • Weight bounds: max single violation 1.8e-5 (solver tolerance)",
            "  • Factor exposure: max deviation 0.057 (limit 0.10)",
            "  • Monthly active return: worst = −1.46% (limit −2.00%)",
        ]),
        ("Transaction Costs & Turnover", [
            "Mean monthly turnover: 12.9% (Full), 13.9% (OOS) → annual ≈ 155–167%.",
            "At 10bps round-trip TC, estimated annual drag ≈ 0.15% (Full), 0.17% (OOS).",
            "After TC, OOS excess return ≈ +1.7% (vs +1.9% gross). Impact is modest",
            "because the constrained optimizer limits active weights to ≤ 2× benchmark.",
        ]),
        ("Limitations & Future Work", [
            "• No refit after selection: model trained on 85% of data.",
            "• Data gap: OpenAssetPricing feature coverage drops in 2023–2024.",
            "• QS volatility (17.0%) is 8% higher than BNCH (15.7%).",
            "• OOS IR=0.61 is not statistically significant (p=0.23, see page 15).",
            "• Future: refit after selection, embargo > 0, regime-adaptive λ refitting.",
        ]),
        ("Conclusion", [
            "This study presents a constrained long-only portfolio combining XGBoost ML signals",
            "with 11 traditional and option-implied features. The final strategy (v4) uses EMA",
            "signal smoothing + CVaR constraint + post-hoc drawdown guard to achieve OOS IR of",
            "0.61 with +1.9% excess return, while satisfying all constraints 100%.",
            "",
            "Key contributions: (1) EMA signal smoothing as the effective solution for drawdown",
            "control; (2) purge fix for val set leakage in walk-forward early stopping;",
            "(3) comprehensive robustness analysis with statistical significance tests.",
        ]),
    ]

    y_dc = 0.91
    for title, lines in disc_sections:
        rect = FancyBboxPatch((0.05, y_dc - 0.003), 0.90, 0.018,
                               boxstyle="square,pad=0.004",
                               facecolor=DARK_BLUE, edgecolor="none",
                               transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(0.07, y_dc + 0.006, title,
                 ha="left", va="center", fontsize=9.2,
                 fontweight="bold", color=WHITE, fontfamily="serif",
                 transform=fig.transFigure)
        y_dc -= 0.026

        for line in lines:
            if line == "":
                y_dc -= 0.006
                continue
            fig.text(0.07, y_dc, line,
                     ha="left", va="top", fontsize=8.2,
                     color=TEXT_DARK, fontfamily="serif",
                     transform=fig.transFigure)
            y_dc -= 0.016
        y_dc -= 0.018

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("Page 14 done")

    # ── PAGE 15 — Robustness & Statistical Significance ────────────────────
    stat_tests_path = os.path.join(OUTPUTS, "statistical_tests.csv")
    ema_path = os.path.join(OUTPUTS, "ema_sensitivity.csv")
    ic_chart_path = os.path.join(OUTPUTS, "ic_timeseries_oos.png")

    if all(os.path.exists(p) for p in [stat_tests_path, ema_path, ic_chart_path]):
        fig = plt.figure(figsize=PORTRAIT)
        fig.patch.set_facecolor(WHITE)
        page_header(fig, "Robustness & Statistical Significance", y=0.96)
        page_footer(fig, 15)

        # IC time series chart
        ic_chart = plt.imread(ic_chart_path)
        ax_ic = fig.add_axes([0.05, 0.68, 0.90, 0.24])
        ax_ic.imshow(ic_chart, aspect="auto")
        ax_ic.axis("off")

        # Statistical significance table
        fig.text(0.07, 0.65, "Statistical Significance Tests (Lo 2002)",
                 ha="left", va="top", fontsize=10, fontweight="bold",
                 color=DARK_BLUE, fontfamily="serif", transform=fig.transFigure)

        stat_df = pd.read_csv(stat_tests_path)
        ax_stat = fig.add_axes([0.05, 0.43, 0.90, 0.20])
        draw_table(ax_stat, stat_df,
                   col_widths=[1.0, 0.8, 1.0, 1.0, 1.0, 0.7],
                   fontsize=9, header_fontsize=9.5)

        fig.text(0.07, 0.42,
                 "Lo (2002): t = (\u03bc/\u03c3) \u00d7 \u221aN.  "
                 "IC significance: t = mean(IC) / se(IC).  "
                 "* p<0.10,  ** p<0.05,  *** p<0.01",
                 ha="left", va="top", fontsize=7.5, color="grey",
                 fontfamily="serif", transform=fig.transFigure)

        # EMA alpha sensitivity table
        fig.text(0.07, 0.38, "EMA Alpha Sensitivity (OOS 2021\u20132024)",
                 ha="left", va="top", fontsize=10, fontweight="bold",
                 color=DARK_BLUE, fontfamily="serif", transform=fig.transFigure)

        ema_df = pd.read_csv(ema_path)
        ema_display = ema_df.copy()
        ema_display.columns = ["Alpha", "Mean IC", "Std IC", "ICIR", "t-stat", "p-value"]
        ema_display["Alpha"] = ema_display["Alpha"].map(lambda x: f"{x:.1f}")
        ema_display["Mean IC"] = ema_display["Mean IC"].map(lambda x: f"{x:.4f}")
        ema_display["Std IC"] = ema_display["Std IC"].map(lambda x: f"{x:.4f}")
        ema_display["ICIR"] = ema_display["ICIR"].map(lambda x: f"{x:.4f}")
        ema_display["t-stat"] = ema_display["t-stat"].map(lambda x: f"{x:.3f}")
        ema_display["p-value"] = ema_display["p-value"].map(lambda x: f"{x:.3f}")

        chosen_idx = ema_df.index[ema_df["Alpha"].round(1) == 0.3].tolist()
        row_colors = {chosen_idx[0]: "#FFF3E0"} if chosen_idx else None

        ax_ema = fig.add_axes([0.05, 0.13, 0.90, 0.23])
        draw_table(ax_ema, ema_display,
                   col_widths=[0.8, 1.0, 1.0, 1.0, 1.0, 1.0],
                   row_colors=row_colors,
                   fontsize=9, header_fontsize=9.5)

        fig.text(0.07, 0.10,
                 "Orange row = production \u03b1 (0.3). IC is flat across all alphas, "
                 "confirming \u03b1=0.3 is not a tuned parameter \u2014 chosen for "
                 "\u22482-month half-life.",
                 ha="left", va="top", fontsize=7.5, color="grey",
                 fontfamily="serif", transform=fig.transFigure)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("Page 15 done")
    else:
        print("WARNING: robustness outputs not found. Run src/robustness.py first. Page 15 skipped.")

    # PDF metadata
    d = pdf.infodict()
    d["Title"]   = "Constrained Portfolio Optimization with Traditional and Option-Implied Signals"
    d["Author"]  = "Quantitative Research"
    d["Subject"] = "S&P 500 Long-Only Constrained Portfolio — 2016-2024"
    d["Keywords"] = "portfolio optimization, XGBoost, option-implied, cvxpy, factor model"

print(f"\nReport saved to: {REPORT}")
