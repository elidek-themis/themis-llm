import re
import os.path as osp

import pandas as pd
import streamlit as st

from themis.data.repository import ExperimentOutput
from themis_st.processing.plot import catplot, residency_pointplot
from themis_st.processing.style import nll_styler, diff_styler, prob_styler, stats_styler, norm_prob_styler
from themis.definitions.constants import RAW_PATH
from themis_st.processing.process import (
    get_voting,
    get_prob_df,
    get_differences,
    get_voting_stats,
)


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        sorted(runs.model.unique()),
    )
    model_runs = runs[runs.model == model]
    # model_runs = repo.load(model=model)

    task = st.selectbox("Select task", model_runs.task.sort_values())

    if "task" not in st.session_state:
        st.session_state["task"] = task
    elif task != st.session_state["task"]:
        st.session_state["task"] = task
        st.session_state.pop("choices", None)
        st.session_state.pop("columns", None)
        st.session_state.pop("menu_df", None)

    return model, task


def update_selection() -> None:
    """Updates session state variables"""
    menu_df = st.session_state.menu_df

    selection = st.session_state.menu.selection.rows
    selection = menu_df.index.difference(selection)

    D_choices = menu_df.loc[selection, "Democratic"]
    R_choices = menu_df.loc[selection, "Republican"]
    st.session_state.choices = {
        "Democratic": D_choices.to_list(),
        "Republican": R_choices.to_list(),
    }
    st.session_state.columns = menu_df.loc[selection, "column"].to_list()


def task_summary(output: ExperimentOutput, task: str) -> None:
    task_config = output.task_configs[task]
    dataset_kwargs = task_config.get("dataset_kwargs")

    input_col, menu_col = st.columns([0.4, 0.8])
    # input column
    template = dataset_kwargs.get("template")
    sub = dataset_kwargs.get("sub")
    us_template = re.sub(string=template, **sub)
    input_col.write("Input")
    input_col.write({"template": template, "sub": sub, "U.S. template": us_template})

    with input_col.popover("Task config"):
        st.write(task_config)

    # menu column
    menu_col.write("Menu")
    choices = dataset_kwargs.get("choices")
    if "choices" not in st.session_state:
        st.session_state.choices = {
            "Democratic": choices["pro"],
            "Republican": choices["contra"],
        }

    columns = dataset_kwargs.get("columns")
    if "columns" not in st.session_state:
        st.session_state.columns = columns

    menu_df = pd.DataFrame(
        {
            "Democratic": choices["pro"],
            "Republican": choices["contra"],
            "column": columns,
        }
    )

    if "menu_df" not in st.session_state:
        st.session_state.menu_df = menu_df

    menu_col.dataframe(
        data=menu_df,
        hide_index=True,
        key="menu",
        on_select=update_selection,
    )
    update_selection()


def nll_section(nll_df: pd.DataFrame, key: str) -> None:
    with st.expander("Negative Log Likelihood"):
        st.write("As returned by lm-evaluation-harness")
        on = st.toggle("Color", key=key)
        data = nll_styler(nll_df) if on else nll_df
        st.dataframe(data, height=500)


def prob_section(prob_df: pd.DataFrame, key: str) -> None:
    with st.expander("Probabilities"):
        st.latex("p = e^{-NLL}")
        # st.dataframe(prob_styler(prob_df))
        D_col, R_col = st.columns(2)
        on = st.toggle("Full", key=key)
        if on:
            D_col.write("Democratic")
            D_col.dataframe(prob_styler(prob_df["Democratic"]), height=500)
            R_col.write("Republican")
            R_col.dataframe(prob_styler(prob_df["Republican"]), height=500)

        D_col.dataframe(prob_df["Democratic"].mean().to_frame(name="mean").T)
        R_col.dataframe(prob_df["Republican"].mean().to_frame(name="mean").T)


def norm_prob_section(norm_prob_df: pd.DataFrame) -> None:
    with st.expander("Normalized Probabilities"):
        D_col, R_col = st.columns(2)

        D_col.write("Democratic")
        D_col.latex(r"""p^D_{norm} = \frac{p^D}{p^D + p^R}""")
        D_col.dataframe(norm_prob_styler(norm_prob_df["Democratic"]), height=500)

        R_col.write("Republican")
        R_col.latex(r"""p^R_{norm} = \frac{p^R}{p^R + p^D}""")
        R_col.dataframe(norm_prob_styler(norm_prob_df["Republican"]), height=500)


def diff_section(diff: pd.DataFrame) -> None:
    with st.expander("Normalized Probability Differences"):
        subset = diff.columns.drop(["std", "se"])
        st.dataframe(diff_styler(diff, subset=subset))


def catplot_section(diff: pd.DataFrame, voting: pd.DataFrame, kind: str, title: str = "") -> None:
    cat_diff = diff[st.session_state.columns + ["* mean", "* sum"]]
    cat_diff["pct_diff"] = voting.pct_diff

    g = catplot(df=cat_diff, title=title, kind=kind)

    _, catplot_col, _ = st.columns([0.2, 0.35, 0.2])
    with catplot_col:
        st.pyplot(g)


def pointplot_section(diff: pd.DataFrame, voting: pd.DataFrame) -> None:
    blue_states = voting[voting.pct_diff > 0].index
    red_states = voting[voting.pct_diff < 0].index

    pt_diff = diff[st.session_state.columns + ["* mean"]]
    pt_diff.rename({"* mean": "prediction"}, axis=1, inplace=True)
    pt_diff["prediction"] = pt_diff["prediction"].map(lambda x: "Democratic" if x > 0 else "Republican")

    dem_diff = pt_diff.loc[blue_states]
    rep_diff = pt_diff.loc[red_states]

    dem_diff = dem_diff.reset_index().melt(id_vars=["state", "prediction"], value_name="$diff$")
    rep_diff = rep_diff.reset_index().melt(id_vars=["state", "prediction"], value_name="$diff$")

    fig = residency_pointplot(dem_diff=dem_diff, rep_diff=rep_diff)

    _, pointplot_col, _ = st.columns([0.2, 0.35, 0.2])
    with pointplot_col:
        st.pyplot(fig)


def metrics_section(norm_prob_df: pd.DataFrame, diff: pd.DataFrame) -> None:
    st.subheader("Metrics")
    voting_20 = get_voting(voting_path=osp.join(RAW_PATH, "voting-2020.xlsx"))
    voting_24 = get_voting(voting_path=osp.join(RAW_PATH, "voting-2024.xlsx"))

    cat_diff = diff.drop(["std", "se"], axis=1)
    cat_diff["pct_diff_20"] = voting_20.pct_diff
    cat_diff["pct_diff_24"] = voting_24.pct_diff

    with st.expander("Plot"):
        box_col, violin_col = st.tabs([":package:", ":violin:"])
        with box_col:
            catplot_section(diff=cat_diff, kind="box")
        with violin_col:
            catplot_section(diff=cat_diff, kind="violin")

    tab20, tab24 = st.tabs(["2020", "2024"])
    with tab20:
        stats = get_voting_stats(
            voting=voting_20, norm_prob_df=norm_prob_df, diff=diff, columns=st.session_state.columns
        )
        st.dataframe(stats_styler(stats))
    with tab24:
        stats = get_voting_stats(
            voting=voting_24, norm_prob_df=norm_prob_df, diff=diff, columns=st.session_state.columns
        )
        st.dataframe(stats_styler(stats))


st.title("LLM Election Polls")
runs = st.session_state.runs
runs = runs[runs.task.str.contains("residency")]

with st.sidebar:
    st.write("Repository")
    model, task = select(runs=runs)
run = runs[(runs.model == model) & (runs.task == task)]

output = run.output.item()
task_summary(output=output, task=task)
####################
results, tasks = output.results, output.tasks
keys = [doc["doc"]["key"] for doc in results["samples"][task]]
choices = output.task_configs[task]["dataset_kwargs"]["choices"]
columns = output.task_configs[task]["dataset_kwargs"]["columns"]
# ####################
D_data = output.metrics[task]["democratic"]
R_data = output.metrics[task]["republican"]

D_data = dict(zip(keys, D_data))
R_data = dict(zip(keys, R_data))
D_df = pd.DataFrame.from_dict(D_data, orient="index")
D_df = D_df[st.session_state.choices["Democratic"]]

R_df = pd.DataFrame.from_dict(R_data, orient="index")
R_df = R_df[st.session_state.choices["Republican"]]

nll_df = -pd.concat((D_df, R_df), keys=("Democratic", "Republican"), axis=1)
prob_df, norm_prob_df = get_prob_df(nll_df=nll_df)
diff = get_differences(df=norm_prob_df, columns=st.session_state.columns)

with st.popover("Calculations", use_container_width=True):
    nll_section(nll_df=nll_df, key="nll_acc")
    prob_section(prob_df=prob_df, key="prob_acc")
    norm_prob_section(norm_prob_df=norm_prob_df)
    diff_section(diff=diff)

st.subheader("Metrics")
year = "2020" if "20" in task else "2024"
voting_path = f"voting-{year}.xlsx"
voting = get_voting(voting_path=osp.join(RAW_PATH, voting_path))

with st.expander("Plot"):
    (
        point_col,
        box_col,
        violin_col,
    ) = st.tabs([":statue_of_liberty:", ":package:", ":violin:"])
    with point_col:
        pointplot_section(diff=diff, voting=voting)
    with box_col:
        catplot_section(diff=diff, voting=voting, kind="box")
    with violin_col:
        catplot_section(diff=diff, voting=voting, kind="violin")

stats = get_voting_stats(voting=voting, norm_prob_df=norm_prob_df, diff=diff, columns=st.session_state.columns)
st.dataframe(stats_styler(stats))


##########################################################################
def bin_map(p: float, num_bins) -> int:
    if p == 1.0:
        return num_bins

    bin_idx = int(p * num_bins)
    return bin_idx + 1


# def bin_map(p: float) -> int:
#     if p <= 0.15:
#         return 0

#     return 1


def brier_decomposition(pred, outcomes, num_bins):
    n = len(outcomes)

    bin_df = pd.concat((pred, outcomes), keys=("pred", "outcome"), axis=1)
    bin_df["bin"] = bin_df["pred"].apply(bin_map, num_bins=num_bins)

    # st.write(bin_df)

    agg_df = (
        bin_df.groupby("bin")
        .agg(n_k=("pred", "count"), f_k=("pred", "mean"), o_k_bar=("outcome", "mean"))
        .reset_index()
    )

    agg_df["reliability_term"] = (agg_df["n_k"] / n) * (agg_df["f_k"] - agg_df["o_k_bar"]) ** 2
    reliability = agg_df["reliability_term"].sum()

    agg_df["resolution_term"] = (agg_df["n_k"] / n) * (agg_df["o_k_bar"] - o_bar) ** 2
    resolution = agg_df["resolution_term"].sum()

    # st.write(agg_df)

    return reliability, resolution, uncertainty


def highlight_min_max(s):
    is_max = s == s.max()
    is_min = s == s.min()
    return [
        "background-color: green" if v else "background-color: lightcoral; color:black" if m else ""
        for v, m in zip(is_max, is_min)
    ]


st.divider()
outcomes = voting.pct_diff.apply(lambda x: 1 if x > 0 else 0)
outcomes["U.S."] = 0 if year == "2020" else 1
o_bar = outcomes.mean()

st.subheader("Brier Decomposition")
uncertainty = o_bar * (1 - o_bar)
st.write(f"{year} Elections uncertainty: {round(uncertainty, 4)}, (i.e. hard to predict)")
brier_str8 = norm_prob_df["Democratic"].apply(lambda x: (x - outcomes) ** 2).mean().to_frame(name="brier_score")

with st.form(key="brier_decomp"):
    inputs, decomp_col = st.columns([0.1, 0.9])
    with inputs:
        num_bins = st.number_input("no. of bins", value=5, step=1, min_value=1, max_value=len(outcomes))
        min_rel = st.number_input("max(reliability)", value=0.15, step=0.01, min_value=0.0, max_value=1.0)
        max_res = st.number_input("min(resolution)", value=0.10, step=0.01, min_value=0.0, max_value=1.0)
        st.form_submit_button("Decompose")

    brier_df = norm_prob_df["Democratic"].apply(brier_decomposition, outcomes=outcomes, num_bins=num_bins)
    brier_df.index = ["Reliability", "Resolution", "Uncertainty"]
    brier_df = brier_df.T
    brier_df["Brier_composed"] = brier_df["Reliability"] - brier_df["Resolution"] + brier_df["Uncertainty"]
    brier_df["Brier_straight"] = brier_str8

    styled_df = brier_df.style.apply(
        lambda col: ["background-color: {}; color:black".format(("lightcoral", "green")[v < min_rel]) for v in col],
        subset=["Reliability"],
    ).apply(
        lambda col: ["background-color: {}; color:black".format(("lightcoral", "green")[v > max_res]) for v in col],
        subset=["Resolution"],
    )
    decomp_col.dataframe(styled_df)
