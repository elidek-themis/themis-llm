import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


def catplot(df: pd.DataFrame, title: str, kind: str) -> sns.FacetGrid:
    df = df.melt(value_name="$P_{diff}$")
    df.rename(columns={"variable": "completion"}, inplace=True)

    g = sns.catplot(
        data=df,
        kind=kind,
        x="completion",
        y="$P_{diff}$",
        hue="completion",
        # col_wrap=3,
        sharey=False,
        height=4,
        aspect=2,
        # gap=0.9
    )

    g.tick_params(axis="x", rotation=90)
    g.figure.suptitle(title)

    return g


def kdeplot(df: pd.DataFrame, title: str) -> sns.FacetGrid:
    ax = sns.kdeplot(df, fill=True, alpha=0.2)
    ax = sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.5))

    return ax


def residency_pointplot(dem_diff: pd.DataFrame, rep_diff: pd.DataFrame, title: str = "") -> plt.Figure:
    fig, (dem_ax, rep_ax) = plt.subplots(ncols=2, figsize=(8, 5))

    sns.pointplot(
        data=dem_diff,
        x="$diff$",
        y="state",
        hue="prediction",
        linestyle="none",
        linewidth=1.25,
        # palette={"Democratic":"#a4c2f4", "Republican": "#ea9999"},
        palette={"Democratic": "blue", "Republican": "red"},
        ax=dem_ax,
        legend=False,
    )
    dem_ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    dem_ax.set_title("Democratic states", fontsize=10)
    dem_ax.set_xticks([-1, 0, 1])
    dem_ax.set_xlabel("$\\overline{\\text{diff}}$")
    dem_ax.set_ylabel("")
    # dem_ax.set_xticklabels([-1, 0, 1])

    sns.pointplot(
        data=rep_diff,
        x="$diff$",
        y="state",
        hue="prediction",
        linestyle="none",
        linewidth=1.25,
        # palette={"Democratic":"#a4c2f4", "Republican": "#ea9999"},
        palette={"Democratic": "blue", "Republican": "red"},
        ax=rep_ax,
        legend=False,
    )
    rep_ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    rep_ax.set_title("Republican states", fontsize=10)
    rep_ax.set_xticks([-1, 0, 1])
    rep_ax.set_xlabel("$\\overline{\\text{diff}}$")
    rep_ax.set_ylabel("")
    rep_ax.yaxis.tick_right()
    rep_ax.yaxis.set_label_position("right")

    plt.tight_layout()

    return fig


def demographic_pointplot(pt_diff) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.pointplot(
        data=pt_diff,
        x="value",
        y="Demographic",
        hue="prediction",
        dodge=False,
        palette={"Democratic": "blue", "Republican": "red"},
        linestyle="none",
        markersize=5,
        linewidth=2,
        ax=ax,
    )

    ax.scatter(
        data=pt_diff,
        x="pct_diff",
        y="Demographic",
        marker="*",
        c="outcome",
        edgecolors="black",
        linewidths=0.8,
        s=45,
        alpha=0.5,
        zorder=2,
    )

    star = {
        "xdata": [0],
        "ydata": [0],
        "marker": "*",
        "color": "black",
        "markersize": 6,
        "linestyle": "None",
        "label": "actual difference",
    }

    circle = {
        "xdata": [0],
        "ydata": [0],
        "marker": "o",
        "color": "black",
        "markersize": 4,
        "linestyle": "None",
        "label": "predicted difference",
    }

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.legend(handles=[Line2D(**circle), Line2D(**star)], loc="upper center", bbox_to_anchor=(0.5, 1.075), ncols=2)
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel("$\\overline{\\text{diff}}$")

    plt.tight_layout()

    return fig
