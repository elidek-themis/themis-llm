import pandas as pd
import seaborn as sns


def catplot(df: pd.DataFrame, title: str, kind: str):
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


def kdeplot(df: pd.DataFrame, title: str):
    ax = sns.kdeplot(df, fill=True, alpha=0.2)
    ax = sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 0.5))

    return ax
