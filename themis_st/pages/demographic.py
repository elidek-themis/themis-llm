import os.path as osp

import pandas as pd
import streamlit as st

from themis.data.repository import ExperimentOutput
from themis_st.processing.plot import catplot, demographic_pointplot
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
        runs.model.unique(),
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
    index = menu_df.index
    selection = st.session_state.menu.selection.rows
    selection = index.difference(selection)
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
    input_col.write("Input")
    input_col.write({"template": template})

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


def catplot_section(diff: pd.DataFrame, kind: str, title: str = "") -> None:
    g = catplot(df=diff, title=title, kind=kind)

    _, catplot_col, _ = st.columns([0.2, 0.35, 0.2])
    with catplot_col:
        st.pyplot(g)


def pointplot_section(diff: pd.DataFrame, dataset: pd.DataFrame) -> None:
    pt_diff = diff[st.session_state.columns + ["* mean"]].reset_index()
    pt_diff.rename({"* mean": "prediction"}, axis=1, inplace=True)
    pt_diff["prediction"] = pt_diff["prediction"].map(lambda x: "Democratic" if x > 0 else "Republican")

    pt_diff["pct_diff"] = dataset["pct_diff"]
    pt_diff["outcome"] = pt_diff["pct_diff"].apply(lambda x: "blue" if x > 0 else "red")

    pt_diff = pt_diff.melt(id_vars=["Demographic", "pct_diff", "prediction", "outcome"])

    fig = demographic_pointplot(pt_diff=pt_diff)

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
runs = runs[runs.task.str.contains("demographic")]

with st.sidebar:
    st.write("Repository")
    model, task = select(runs=runs)
run = runs[(runs.model == model) & (runs.task == task)]

output = run.output.item()
task_summary(output=output, task=task)

##############
samples = [x["doc"] for x in output.results["samples"][task]]
dataset = pd.DataFrame(samples)[["demographic", "group", "pct"]]

blue_pct = dataset.pct.apply(lambda pct: pct["blue_pct"])
red_pct = dataset.pct.apply(lambda pct: pct["red_pct"])
dataset.drop("pct", axis=1, inplace=True)

dataset["blue_pct"] = blue_pct / (blue_pct + red_pct)
dataset["red_pct"] = red_pct / (blue_pct + red_pct)

dataset["pct_diff"] = dataset["blue_pct"] - dataset["red_pct"]
dataset.set_index(["demographic"], inplace=True)
dataset.loc["LGBT", "group"] = dataset.loc["LGBT", "group"].str.cat((" LGBT", " LGBT"))
dataset = dataset.reset_index(drop=True).rename({"group": "Demographic"}, axis=1)

keys = dataset["Demographic"]
choices = output.task_configs[task]["dataset_kwargs"]["choices"]
columns = output.task_configs[task]["dataset_kwargs"]["columns"]
# ####################
D_data = output.metrics[task]["democratic"]
D_df = pd.concat((keys, pd.DataFrame(D_data)), axis=1).set_index("Demographic")
D_df = D_df[st.session_state.choices["Democratic"]]

R_data = output.metrics[task]["republican"]
R_df = pd.concat((keys, pd.DataFrame(R_data)), axis=1).set_index("Demographic")
R_df = R_df[st.session_state.choices["Republican"]]

nll_df = -pd.concat((D_df, R_df), keys=("Democratic", "Republican"), axis=1)
prob_df, norm_prob_df = get_prob_df(nll_df=nll_df)
diff = get_differences(df=norm_prob_df, columns=st.session_state.columns)

with st.popover("Calculations", use_container_width=True):
    nll_section(nll_df=nll_df, key="nll_acc")
    prob_section(prob_df=prob_df, key="prob_acc")
    norm_prob_section(norm_prob_df=norm_prob_df)
    diff_section(diff=diff)

##########################
pointplot_section(diff=diff, dataset=dataset)
##########################
st.subheader("Predictions")
ground_truth = dataset.set_index(["Demographic"])

options = st.session_state.columns + ["* sum", "* mean"]
completion = st.selectbox("completion", options=options, index=options.index("* mean"))
idx = options.index(completion)
d_prob = norm_prob_df["Democratic"].iloc[:, idx]
r_prob = norm_prob_df["Republican"].iloc[:, idx]

pred_df = pd.concat((d_prob, r_prob), axis=1)
pred_df["pred_pct"] = pred_df.iloc[:, -2] - pred_df.iloc[:, -1]
pred_df = pd.concat((ground_truth, pred_df), axis=1)

fn = lambda x: "background-color: {}; color:black".format(("#ea9999", "#a4c2f4")[x > 0])
st.dataframe(pred_df.style.map(func=fn, subset=pd.IndexSlice[slice(None), ["pct_diff", "pred_pct"]]), height=700)
