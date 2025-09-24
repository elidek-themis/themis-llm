import copy

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
    df = pd.DataFrame(
        data=[(s["doc"]["bias_type"], s["pct_stereotype"]) for s in samples], columns=["bias_type", "pct_stereotype"]
    )

    return df.groupby("bias_type")


def get_pie_figure(samples: list, model: str) -> plt.Figure:
    def autopct_fmt(values):
        def fmt(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))
            return f"{count}"

        return fmt

    colors = {"no-stereo": "#35811e58", "stereo": "#a31919c0"}

    grouped = get_groups(samples=samples)

    fig, ax = plt.subplots(ncols=len(grouped), figsize=(17.5, 4))
    # fig.suptitle("Model", fontsize=20)

    for i, (bias_type, group) in enumerate(grouped):
        counts = group["pct_stereotype"].value_counts()  # .sort_index()

        ax[i].pie(
            counts,
            autopct=autopct_fmt(counts),
            startangle=90,
            colors=[colors["stereo"], colors["no-stereo"]],
        )
        ax[i].set_title(bias_type)
        plt.tight_layout()

    legend_labels = [
        Patch(facecolor=colors["stereo"], label="stereo"),
        Patch(facecolor=colors["no-stereo"], label="no-stereo"),
    ]
    fig.legend(handles=legend_labels, loc="upper left")
    fig.suptitle(t=model)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.75, right=0.85)

    return fig


def get_b_vs_it_figure(df_base: pd.DataFrame, df_it: pd.DataFrame, columns: list) -> sns.FacetGrid:
    df_base, df_it = (df.abs() for df in (df_base, df_it))

    df_base.index = df_base.index.str.replace("-pt", "")
    df_it.index = df_it.index.str.replace("-it", "").str.replace("-Instruct", "")

    df_base["model"], df_it["model"] = df_base.index, df_it.index
    df_base["variant"], df_it["variant"] = "Base", "Instruct"

    df_all = pd.concat([df_base, df_it], axis=0)
    df_all = df_all[["model", "variant"] + columns]
    df_melted = df_all.melt(id_vars=["model", "variant"], var_name="metric", value_name="value")

    g = sns.catplot(data=df_melted, x="model", y="value", hue="variant", col="metric", kind="bar", sharey=False)

    max_ll = df_all[["likelihood_1", "likelihood_2"]].max().max()
    *_, ll_1, ll_2 = g.axes.flatten()
    ll_1.set_ylim(0, max_ll * 1.1)
    ll_2.set_ylim(0, max_ll * 1.1)

    g.set_titles("{col_name}")
    g.set_axis_labels("model", "value")
    g.tight_layout()

    return g


def sample_task_configs(repo: pd.DataFrame):
    data, *_ = repo.sample(1).data.values
    task_configs = copy.deepcopy(data.task_configs)

    for task in task_configs:
        task_configs[task].pop("metadata")

    return task_configs


@st.fragment()
def task_report(repo: pd.DataFrame):
    task_configs = sample_task_configs(repo=repo)  # maybe cache this
    configs = pd.DataFrame.from_dict({"config": task_configs})
    event = st.dataframe(configs, selection_mode="single-row", on_select="rerun")
    selection = event.selection.rows
    if selection:
        selected_task = configs.index[selection].item()
        st.write(selected_task)
        task_metrics = get_metrics(df=repo, task=selected_task).drop("alias", axis=1)
        st_df(df=task_metrics, hide_index=False, round=4)
        for _, row in repo.iterrows():
            model = row["model"]
            samples = list(row["data"].samples(selected_task))
            fig = get_pie_figure(samples=samples, model=model)
            st.pyplot(fig)


def table_summary(repo):
    for _, row in repo.iterrows():
        model = row["model"]
        results = row["data"].results
        st.write(f"#### {model}")
        table = make_table(results)
        st.write(table)


st.header("CrowS-Pairs")

if "repo" not in st.session_state:
    with st.spinner("Loading repository..."):
        st.session_state.repo = get_repo()
        st.session_state.repo.model = st.session_state.repo.model.apply(lambda x: x.split("__")[1])

base_group = "crows_pairs_base"
base_repo = st.session_state.repo[st.session_state.repo.task == base_group]
it_group = "crows_pairs_instruct"
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
        table_summary(repo=it_repo)

_, b_v_it, _ = st.columns((0.2, 0.6, 0.2))
with b_v_it:
    columns = ["pct_stereotype", "likelihood_diff", "likelihood_1", "likelihood_2"]
    g = get_b_vs_it_figure(df_base=base_metrics, df_it=it_metrics, columns=columns)
    st.pyplot(g.figure)


st.divider()
base_col, it_col = st.columns((0.5, 0.5))


with base_col:
    task_report(repo=base_repo)

with it_col:
    task_report(repo=it_repo)
