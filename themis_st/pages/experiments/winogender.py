import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from lm_eval.utils import make_table
from matplotlib.patches import Patch

from themis.data.repository import get_repo


def st_df(df: pd.DataFrame, hide_index: bool = True, round: int | None = None) -> None:
    """Avoid arrow exception"""
    if round:
        df = df.round(round)
    st.dataframe(df.astype(str), hide_index=hide_index)


def get_metrics(df: pd.DataFrame, task: str) -> pd.DataFrame:
    metrics = df.data.apply(lambda x: x.metrics[task])
    metrics = pd.json_normalize(metrics)
    metrics.index = df.model.tolist()

    return metrics


def get_groups(samples: list) -> pd.DataFrame:
    df = pd.DataFrame(data=[(s["doc"]["bias_type"], s["icat"]) for s in samples], columns=["bias_type", "icat"])

    return df.groupby("bias_type")


def get_pie_figure(samples: list, model: str) -> plt.Figure:
    colors = {"icat": "#35811e58", "1-icat": "#a31919c0"}

    grouped = get_groups(samples=samples)

    fig, ax = plt.subplots(ncols=len(grouped), figsize=(15, 3))
    # fig.suptitle("Model", fontsize=20)

    for i, (bias_type, group) in enumerate(grouped):
        mean_icat = group["icat"].mean()
        counts = [mean_icat, 1 - mean_icat]

        ax[i].pie(
            counts,
            autopct="%1.1f%%",
            startangle=90,
            colors=[colors["icat"], colors["1-icat"]],
        )
        ax[i].set_title(bias_type)
        plt.tight_layout()

    legend_labels = [
        Patch(facecolor=colors["icat"], label="icat"),
        Patch(facecolor=colors["1-icat"], label="1-icat"),
    ]
    fig.legend(handles=legend_labels, loc="upper left")
    fig.suptitle(t=model)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.75, right=0.85)

    return fig


def get_b_vs_it_figure(df_base: pd.DataFrame, df_it: pd.DataFrame, columns: list) -> sns.FacetGrid:
    df_base.index = df_base.index.str.replace("-pt", "")
    df_it.index = df_it.index.str.replace("-it", "").str.replace("-Instruct", "")

    df_base["model"], df_it["model"] = df_base.index, df_it.index
    df_base["variant"], df_it["variant"] = "Base", "Instruct"

    df_all = pd.concat([df_base, df_it], axis=0)
    df_all = df_all[["model", "variant"] + columns]
    df_melted = df_all.melt(id_vars=["model", "variant"], var_name="metric", value_name="value")

    g = sns.catplot(data=df_melted, x="model", y="value", hue="variant", col="metric", kind="bar", sharey=False)

    # g.set_titles("{col_name}")
    # g.set_axis_labels("model", "value")
    g.tight_layout()

    return g


def table_summary(repo):
    for _, row in repo.iterrows():
        model = row["model"]
        results = row["data"].results
        st.write(f"#### {model}")
        table = make_table(results)
        st.write(table)


st.header("Winogender")

if "repo" not in st.session_state:
    with st.spinner("Loading repository..."):
        st.session_state.repo = get_repo()
        st.session_state.repo.model = st.session_state.repo.model.apply(lambda x: x.split("__")[1])


options = [
    "winogender_all",
    "winogender_male",
    "winogender_female",
    "winogender_neutral",
    "winogender_gotcha_male",
    "winogender_gotcha_female",
]
task_group = st.selectbox(label="Task", options=options, index=0)
base_group = task_group + "_base"
it_group = task_group + "_it"

base_repo = st.session_state.repo[st.session_state.repo.task == base_group]
it_repo = st.session_state.repo[st.session_state.repo.task == it_group]

base_col, it_col = st.columns((0.5, 0.5))

with base_col:
    st.write("### Base", unsafe_allow_html=True)
    base_metrics = get_metrics(df=base_repo, task=base_group)
    st_df(base_metrics, hide_index=False, round=4)
    with st.popover(label="Table summary", use_container_width=True):
        table_summary(repo=base_repo)

with it_col:
    st.write("### Instruct", unsafe_allow_html=True)
    it_metrics = get_metrics(df=it_repo, task=it_group)
    st_df(it_metrics, hide_index=False, round=4)
    with st.popover(label="Table summary", use_container_width=True):
        table_summary(repo=base_repo)

_, b_v_it, _ = st.columns((0.2, 0.6, 0.2))
with b_v_it:
    g = get_b_vs_it_figure(df_base=base_metrics, df_it=it_metrics, columns=["acc", "likelihood_diff"])
    # _, fig_col, _ = st.columns((0.2, 0.6, 0.2))
    st.pyplot(g.figure)


# g = sns.catplot(kind="bar", data=repo, y="acc", hue="model", col="task", col_wrap=3, sharex=True)
