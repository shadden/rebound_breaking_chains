#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive


def fetch_exoplanet_data(cache_file: str, refresh: bool) -> pd.DataFrame:
    """
    Load cached CSV if present and refresh==False; otherwise query the
    NASA Exoplanet Archive and write the CSV cache.
    We intentionally query *all* transiting planets and filter by radius
    per-system locally to ensure the 'all planets < 6 Re' constraint is correct.
    """
    if (not refresh) and os.path.exists(cache_file):
        print(f"[info] Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)

    print("[info] Querying NASA Exoplanet Archive (pscomppars, transiting only)...")
    tab = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name,hostname,pl_orbper,pl_rade,tran_flag",
        where="tran_flag = 1",
        cache=False,
    )
    df = tab.to_pandas()

    # Save raw query for transparency/reproducibility
    df.to_csv(cache_file, index=False)
    print(f"[info] Saved fresh query to {cache_file} ({len(df)} rows).")
    return df


def filter_compact_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only systems (grouped by hostname) that:
      - have at least 3 transiting planets,
      - have no missing orbital periods or radii,
      - and ALL planets in the system have pl_rade < 6 Earth radii.
    """
    def _valid_group(g: pd.DataFrame) -> bool:
        if len(g) < 3:
            return False
        if g["pl_orbper"].isnull().any() or g["pl_rade"].isnull().any():
            return False
        return (g["pl_rade"] < 6).all()

    grouped = df.groupby("hostname", sort=False)
    filtered = grouped.filter(_valid_group)
    return filtered


def compute_period_ratio_points(filtered: pd.DataFrame):
    """
    For each system (sorted by period), for each internal planet i (1..N-2):
      x = P_i / P_{i-1}
      y = P_i / P_{i+1}
    Returns x_vals, y_vals (np.ndarray), and the grouped object for reuse.
    """
    systems = filtered.groupby("hostname", sort=False)
    x_vals, y_vals = [], []
    for _, group in systems:
        g = group.sort_values("pl_orbper")
        P = g["pl_orbper"].to_numpy()
        if len(P) < 3:
            continue
        for i in range(1, len(P) - 1):
            x_vals.append(P[i] / P[i - 1])
            y_vals.append(P[i] / P[i + 1])
    return np.array(x_vals), np.array(y_vals), systems

def compute_highlight_points(systems, highlight_ordered_list):
    """
    Return an OrderedDict: system_name -> [(x1, y1), (x2, y2), ...],
    preserving the input order.
    """
    highlight_data = OrderedDict()
    system_map = {name: group for name, group in systems}

    for name in highlight_ordered_list:
        if name not in system_map:
            continue
        group = system_map[name].sort_values("pl_orbper")
        P = group["pl_orbper"].to_numpy()
        if len(P) < 3:
            continue
        xy = [(P[i] / P[i - 1], P[i] / P[i + 1]) for i in range(1, len(P) - 1)]
        highlight_data[name] = xy

    return highlight_data


def add_resonance_guides(ax, j_min=2, j_max=5):
    """
    Draw vertical (x = j/(j-1)), horizontal (y = (j-1)/j), and optional diagonal
    guide lines for j in [j_min, j_max], with white-boxed 'j:j-1' labels.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xvals = np.linspace(*xlim, 500)

    for j in range(j_min, j_max + 1):
        # Horizontal line: y = (j-1)/j  (corresponds to P_i / P_{i+1})
        yres = (j - 1) / j
        ax.axhline(yres, color='k', lw=1)
        ax.text(
            xlim[0] + 0.01, yres + 0.005, f"{j}:{j-1}",
            va='bottom', ha='left', fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
        )

        # Vertical line: x = j/(j-1)  (corresponds to P_i / P_{i-1})
        xres = j / (j - 1)
        ax.axvline(xres, color='k', lw=1)
        ax.text(
            xres + 0.01, ylim[0] + 0.01, f"{j}:{j-1}",
            va='bottom', ha='left', rotation=90, fontsize=10,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
        )

        # Optional diagonal family (same as in your earlier code)
        for k in range(max(2, j - 1), j + 2):
            ydiag = (1 - j) * xvals / k + (k - 1 + j) / k
            ax.plot(xvals, ydiag, ls='--', color='gray', lw=1)

import itertools
import matplotlib.colors as mcolors

import itertools
import matplotlib.colors as mcolors
GOLDEN = 0.5 * (1 + np.sqrt(5))
def plot_period_ratio_plane(x_vals, y_vals,
                            highlight_dict=None,
                            xlim=(1.2, 2.2), ylim=(1/2.2, 1/1.2),
                            save_path=None):
    fig, ax = plt.subplots(figsize=(6 * GOLDEN, 6 ))
    ax.scatter(x_vals, y_vals, color='red', s=5, zorder=99)

    # Unique color per highlighted system
    if highlight_dict:
        color_cycle = itertools.cycle(plt.get_cmap("tab10").colors)
        for system, xy_list in highlight_dict.items():
            xs, ys = zip(*xy_list)
            color = next(color_cycle)
            ax.scatter(xs, ys, s=30, color=color, edgecolor='k', label=system, zorder=100)

    ax.set_xlabel(r'$P_i / P_{i-1}$')
    ax.set_ylabel(r'$P_i / P_{i+1}$')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    add_resonance_guides(ax, j_min=2, j_max=5)

    ax.set_title("Period Ratio Plane for Multi-transiting Systems")

    
    ax.legend(loc='best', fontsize=9)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[info] Saved figure to {save_path}")
    else:
        plt.show()



def parse_args():
    p = argparse.ArgumentParser(
        description="Plot P_i/P_{i-1} vs P_i/P_{i+1} for compact multi-planet transiting systems (all R < 6 R_earth)."
    )
    p.add_argument(
        "--cache-file",
        default="exoplanet_transiting_pscomppars.csv",
        help="Path to CSV cache for Exoplanet Archive query (default: %(default)s)"
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force a fresh query to the Exoplanet Archive, overwriting the cache."
    )
    p.add_argument(
        "--highlight",
        default="Kepler-223,TOI-178,HD 110067",
        help="Comma-separated hostnames to highlight (default: %(default)s)."
    )
    p.add_argument(
        "--save-plot",
        default="",
        help="If provided, save the plot to this path instead of showing it (e.g., plot.png or plot.pdf)."
    )
    p.add_argument(
        "--xlim",
        default="1.2,2.2",
        help="X-axis limits as 'xmin,xmax' (default: 1.2,2.2)."
    )
    p.add_argument(
        "--ylim",
        default=f"{1/2.2:.5f},{1/1.2:.5f}",
        help=f"Y-axis limits as 'ymin,ymax' (default: {1/2.2:.5f},{1/1.2:.5f})."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Parse limits
    xlim = tuple(float(v) for v in args.xlim.split(","))
    ylim = tuple(float(v) for v in args.ylim.split(","))

    # Fetch + cache data, then filter
    df = fetch_exoplanet_data(args.cache_file, args.refresh)
    filtered = filter_compact_multiples(df)

    if filtered.empty:
        print("[warn] No systems met the filter criteria (>=3 planets, all R<6 Re, no missing periods/radii).")
        return

    # Compute points
    x_vals, y_vals, systems = compute_period_ratio_points(filtered)

    # Highlights
    from collections import OrderedDict
    highlight_hosts = [h.strip() for h in args.highlight.split(",") if h.strip()]

    highlight_dict = compute_highlight_points(systems, highlight_hosts)
    print(highlight_hosts)
    # Plot
    save_path = args.save_plot if args.save_plot else None

    plot_period_ratio_plane(
        x_vals, y_vals,
        highlight_dict=highlight_dict,
        xlim=xlim, ylim=ylim,
        save_path=save_path
    )

if __name__ == "__main__":
    main()
