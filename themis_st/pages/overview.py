import re
import os.path as osp

import pandas as pd
import streamlit as st

from themis.data.repository import Repository, ExperimentOutput
from themis_st.processing.plot import catplot
from themis_st.processing.style import nll_styler, diff_styler, prob_styler, stats_styler, norm_prob_styler
from themis.definitions.constants import RAW_PATH
from themis_st.processing.process import (
    ElectionResults,
    get_nll_df,
    get_voting,
    get_prob_df,
    get_differences,
    get_voting_stats,
)


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        runs.model.unique(),
    )
    model_runs = repo.load(model=model)

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
    index = menu_df.index
    selection = st.session_state.menu.selection.rows
    selection = index.difference(selection)
    D_choices = menu_df.loc[selection, "Democratic"]
    R_choices = menu_df.loc[selection, "Republican"]
    st.session_state.choices = D_choices.to_list() + R_choices.to_list()
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

    # menu column
    menu_col.write("Menu")
    choices = dataset_kwargs.get("choices")
    if "choices" not in st.session_state:
        st.session_state.choices = choices

    no_choices = int(len(choices) / 2)
    D_choices, R_choices = choices[:no_choices], choices[no_choices:]
    columns = dataset_kwargs.get("columns")
    if "columns" not in st.session_state:
        st.session_state.columns = columns

    menu_df = pd.DataFrame(
        {
            "Democratic": D_choices,
            "Republican": R_choices,
            "column": columns,
        }
    )
    # if "menu_df" not in st.session_state:
    st.session_state.menu_df = menu_df

    menu_col.dataframe(
        data=menu_df,
        hide_index=True,
        key="menu",
        on_select=update_selection,
    )


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


def catplot_section(diff: pd.DataFrame, kind: str, title: str = "") -> None:
    g = catplot(df=diff, title=title, kind=kind)

    _, catplot_col, _ = st.columns([0.2, 0.35, 0.2])
    with catplot_col:
        st.pyplot(g)


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
        error_df, stats = get_voting_stats(
            voting=voting_20, norm_prob_df=norm_prob_df, diff=diff, columns=st.session_state.columns
        )
        st.dataframe(stats_styler(stats))
    with tab24:
        error_df, stats = get_voting_stats(
            voting=voting_24, norm_prob_df=norm_prob_df, diff=diff, columns=st.session_state.columns
        )
        st.dataframe(stats_styler(stats))

    return (
        error_df,
        voting_24,
    )


st.title("LLM Election Polls")
repo = Repository()
runs = repo.runs

with st.sidebar:
    st.write("Repository")
    model, task = select(runs=runs)
run = repo.load(model=model, task=task)

output = run.output.item()
task_summary(output=output, task=task)

results = ElectionResults(output=output)

data = results.metrics[task]["acc"]
nll_df = get_nll_df(data=data, index=results.keys, columns=results.choices[task], use_cols=st.session_state.choices)
nll_section(nll_df=nll_df, key="nll_acc")

prob_df, norm_prob_df = get_prob_df(nll_df=nll_df)
prob_section(prob_df=prob_df, key="prob_acc")
norm_prob_section(norm_prob_df=norm_prob_df)
diff = get_differences(df=norm_prob_df, columns=st.session_state.columns)
diff_section(diff=diff)
error_df, voting = metrics_section(norm_prob_df=norm_prob_df, diff=diff)
