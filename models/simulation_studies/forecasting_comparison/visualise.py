#!/usr/bin/env python3
"""
plot_means.py

Usage examples
--------------
Basic (auto-labels “row 1”, “row 2”, …):
    python plot_means.py means.csv

With SE values:
    python plot_means.py means.csv --se_csv se.csv

Specify legend labels (comma-separated):
    python plot_means.py means.csv --labels "Alpha,Beta,Gamma"

Save to a specific PDF file:
    python plot_means.py means.csv --labels "A,B,C" -o myplot.pdf
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import time
import numpy as np
import plotly.colors as pc  

# PRESET_LABELS = [
#     r"r$\mathcal{N}(0,\Sigma)$",
#     r"r$t_{7}(0,\Sigma)$",
#     r"r$t_{5}(0,\Sigma)$",
#     r"r$t_{3}(0,\Sigma)$",
# ]

PRESET_LABELS = ["NIRVAR","BayesianVAR"] 

# ── Layout (as provided) ──────────────────────────────────────────────────────
layout = go.Layout(
    yaxis=dict(showline=True, linewidth=1, linecolor="black",
               ticks="outside", mirror=True),
    xaxis=dict(showline=True, linewidth=1, linecolor="black",
               ticks="outside", mirror=True, automargin=True),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_family="Serif",
    font_size=14,
    margin=dict(l=5, r=5, t=5, b=5),
    width=500,
    height=350
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_means_csv(path: Path) -> tuple[pd.DataFrame, list]:
    """
    Read means CSV, return (means_df, x_vals).
      • x_vals are taken from the first *row* (excluding the first column).
      • means_df contains numeric data starting from row 2 / col 2.
    """
    df = pd.read_csv(path, header=None)

    # X-axis: first row, columns 1..end
    x_vals_raw = df.iloc[0, 1:].tolist()
    # Convert to numeric if possible, otherwise leave as strings
    x_vals = pd.to_numeric(x_vals_raw, errors="ignore").tolist()

    # Means: rows 1..end, columns 1..end
    means_df = df.iloc[1:, 1:].astype(float)
    return means_df, x_vals


def load_se_csv(path: Path) -> pd.DataFrame:
    """
    Read SE CSV, drop first row/col (matching means), and return numeric DataFrame.
    """
    df = pd.read_csv(path, header=None)
    return df.iloc[1:, 1:].astype(float)


def build_figure(means: pd.DataFrame,
                 se: pd.DataFrame | None,
                 labels: list[str] | None,
                 x_vals: list) -> go.Figure:
    """Plot mean lines; if SE given, add shaded ribbons instead of bars."""
    fig = go.Figure(layout=layout)
    default_colors = pio.templates["plotly"].layout.colorway

    for i, (_, row) in enumerate(means.iterrows()):
        name   = labels[i] if labels and i < len(labels) else f"row {i+1}"
        color  = default_colors[i % len(default_colors)]

        # 1) mean line (always)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=row.values,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            marker=dict(color=color)
        ))

        # 2) ribbon if we have SE data
        if se is not None:
            upper = row.values + se.iloc[i].values/(np.sqrt(100*50))
            lower = row.values - se.iloc[i].values/(np.sqrt(100*50))

            rgb = pc.hex_to_rgb(color)              # e.g. (31, 119, 180)
            fillcolor = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.20)"


            # plot upper bound (invisible line) …
            fig.add_trace(go.Scatter(
                x=x_vals, y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))
            # … then lower bound, filling *to next y* (creates the band)
            fig.add_trace(go.Scatter(
                x=x_vals, y=lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo="skip"
            ))

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean values (and optional SEs) from CSV files."
    )
    parser.add_argument("means_csv", type=Path, help="CSV of mean values")
    parser.add_argument("--se_csv", type=Path,
                        help="CSV of standard errors")
    parser.add_argument("--labels", type=str,
                        help="Comma-separated legend labels for each row")
    parser.add_argument("-o", "--output", default="plot.pdf",
                        help="PDF file to write (default: plot.pdf)")
    args = parser.parse_args()

    # ── Load means (and x-axis) ────────────────────────────────────────────────
    means_df, x_vals = load_means_csv(args.means_csv)

    # ── Load SE file if provided ───────────────────────────────────────────────
    se_df = None
    if args.se_csv:
        se_df = load_se_csv(args.se_csv)
        if se_df.shape != means_df.shape:
            sys.exit("Error: means and SE CSVs differ in shape after trimming.")

    # ── Parse labels ───────────────────────────────────────────────────────────
    labels = PRESET_LABELS.copy()  
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split(",")]
        if len(labels) != means_df.shape[0]:
            sys.exit(
                f"Error: You supplied {len(labels)} labels but "
                f"{means_df.shape[0]} rows will be plotted."
            )

    # ── Build & save figure ────────────────────────────────────────────────────
    fig = build_figure(means_df, se_df, labels, x_vals)
    fig.update_xaxes(
    tickmode="array",
    tickvals=x_vals,                     # positions of the ticks
    ticktext=[int(v) for v in x_vals]    # what to print under each tick
    )

    fig.update_layout(
    xaxis_title=r"$N$",   # LaTeX-style math is OK in Plotly titles
    yaxis_title="MSPE"
    )
    # Static image export (requires the kaleido package)
    pio.write_image(fig, file=args.output)   # filetype inferred from ".pdf"
    time.sleep(1)  # Ensure file is written before exit
    pio.write_image(fig, file=args.output)   # filetype inferred from ".pdf"


if __name__ == "__main__":
    main()
