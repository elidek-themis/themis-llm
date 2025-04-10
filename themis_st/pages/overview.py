import re
import pandas as pd
import streamlit as st
import os.path as osp

from themis.data.repository import ExperimentOutput, Repository
from themis.definitions.constants import RAW_PATH

from themis_st.processing.plot import *
from themis_st.processing.process import *
from themis_st.processing.style import *


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        runs.model.unique(),
    )
    model_runs = repo.load(model=model)

    task = st.selectbox(
        "Select task",
        model_runs.task.sort_values()    
    )
    
    if "task" not in st.session_state:
        st.session_state["task"] = task
    else:
        if task != st.session_state["task"]:
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
    
    input_col, menu_col = st.columns([.4, .8])
    # input column
    template = dataset_kwargs.get("template")
    sub = dataset_kwargs.get("sub")
    us_template = re.sub(string=template, **sub)
    input_col.write("Input")
    input_col.write({"template": template, "sub": sub, "U.S. template": us_template})
    
    # menu column
    menu_col.write("Menu")
    choices = dataset_kwargs.get("choices")
    if 'choices' not in st.session_state:
        st.session_state.choices = choices

    no_choices = int(len(choices)/2)
    D_choices, R_choices = choices[:no_choices], choices[no_choices:]
    columns = dataset_kwargs.get("columns")
    if 'columns' not in st.session_state:
        st.session_state.columns = columns

    menu_df = pd.DataFrame(
        {"Democratic": D_choices,
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

def catplot_section(diff: pd.DataFrame, kind: str, title: str="") -> None:
    g = catplot(df=diff, title=title, kind=kind)
    
    _, catplot_col, _ = st.columns([.2, .35, .2])
    with catplot_col:
        st.pyplot(g)

def metrics_section(norm_prob_df: pd.DataFrame, diff: pd.DataFrame) -> None:
    st.subheader("Metrics")
    voting_20 = get_voting(voting_path=osp.join(RAW_PATH, "voting-2020.xlsx"))
    voting_24 = get_voting(voting_path=osp.join(RAW_PATH,"voting-2024.xlsx"))
    
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
            voting=voting_20,
            norm_prob_df=norm_prob_df,
            diff=diff,
            columns=st.session_state.columns
        )
        st.dataframe(stats_styler(stats))
    with tab24:    
        error_df, stats = get_voting_stats(
            voting=voting_24,
            norm_prob_df=norm_prob_df,
            diff=diff,
            columns=st.session_state.columns
        )
        st.dataframe(stats_styler(stats))
    
    return error_df, voting_24, 


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
nll_df = get_nll_df(
    data=data,
    index=results.keys,
    columns=results.choices[task],
    use_cols=st.session_state.choices
)
nll_section(nll_df=nll_df, key="nll_acc")

prob_df, norm_prob_df = get_prob_df(nll_df=nll_df)
prob_section(prob_df=prob_df, key="prob_acc")
norm_prob_section(norm_prob_df=norm_prob_df)
diff = get_differences(df=norm_prob_df, columns=st.session_state.columns)
diff_section(diff=diff)
error_df, voting = metrics_section(norm_prob_df=norm_prob_df, diff=diff)

import math

electoral_votes = {
    "Alabama": 9, "Alaska": 3, "Arizona": 11, "Arkansas": 6, "California": 54,
    "Colorado": 10, "Connecticut": 7, "Delaware": 3, "Florida": 30, "Georgia": 16,
    "Hawaii": 4, "Idaho": 4, "Illinois": 19, "Indiana": 11, "Iowa": 6,
    "Kansas": 6, "Kentucky": 8, "Louisiana": 8, "Maine": 4, "Maryland": 10,
    "Massachusetts": 11, "Michigan": 15, "Minnesota": 10, "Mississippi": 6, "Missouri": 10,
    "Montana": 4, "Nebraska": 5, "Nevada": 6, "New Hampshire": 4, "New Jersey": 14,
    "New Mexico": 5, "New York": 28, "North Carolina": 16, "North Dakota": 3, "Ohio": 17,
    "Oklahoma": 7, "Oregon": 8, "Pennsylvania": 19, "Rhode Island": 4, "South Carolina": 9,
    "South Dakota": 3, "Tennessee": 11, "Texas": 40, "Utah": 6, "Vermont": 3,
    "Virginia": 13, "Washington": 12, "West Virginia": 4, "Wisconsin": 10, "Wyoming": 3,
    "District of Columbia": 3
}

st.divider()
columns = st.session_state.columns + ["* sum", "* mean"]
blue_err = norm_prob_df.drop("U.S.")["Democratic"].\
    apply(lambda x: (voting.blue_pct - x).abs())
blue_err = blue_err.loc[voting.pct_diff > 0]
blue_err.columns = columns

red_err = norm_prob_df.drop("U.S.")["Republican"].\
    apply(lambda x: (voting.red_pct - x).abs())
red_err = red_err.loc[voting.pct_diff < 0]
red_err.columns = columns

abs_error_df = pd.concat(objs=(red_err, blue_err)).sort_index()
st.dataframe(abs_error_df["* mean"])
st.divider()
st.write("### nDCG@")

abs_error_t = pd.Series(electoral_votes).to_frame(name="EV")
abs_error_t["abs_error"] = abs_error_df["* mean"]
abs_error_t = abs_error_t.sort_values(by="EV", ascending=False)
abs_error_t.insert(0, "rank", range(1, len(abs_error_t) + 1))

abs_error_t["abs_error_i"] = abs_error_t.apply(
    lambda i: i["abs_error"]/math.log2(i["rank"]+1),
    axis=1
)

error_t = pd.Series(electoral_votes).to_frame(name="EV")
error_t = error_t.sort_values(by="EV", ascending=False)
error_t.insert(0, "rank", range(1, len(error_t) + 1))

error_t["abs_error_mean"] = abs_error_df["* mean"]
error_t = error_t.sort_values(by="EV", ascending=False)

error_t["abs_error_mean_i"] = error_t.apply(
    lambda i: i["abs_error_mean"]/math.log2(i["rank"]+1),
    axis=1
)

error_t["num_i"] = error_t.apply(
    lambda i: ((1-i["abs_error_mean"]))/math.log2(i["rank"]+1),
    axis=1
)
error_t["DCG"] = error_t["num_i"].cumsum()

error_t["den_i"] = error_t["rank"].apply(lambda i: (1)/math.log2(i+1))
error_t["iDCG"] = error_t["den_i"].cumsum()

idx = [f"NDCG@{i+1}" for i in range(51)]
with st.expander("NDCG Calculations"):
    st.dataframe(error_t)

de = error_t["abs_error_mean"].cumsum()
dce = error_t["abs_error_mean_i"].cumsum()
ndcg = error_t["DCG"] / error_t["iDCG"]

cum_sum_df = pd.DataFrame(data=zip(de,dce,ndcg), columns=["CE", "DCE", "nDCG"])

with st.expander("Cumulative Sums"):
    st.dataframe(cum_sum_df)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()

cum_sum_df[["CE", "DCE"]].plot(ax=ax1, legend=True)
# ax1.set_ylabel("DE & DCE values", color="black")

# Create secondary y-axis for nDCG
ax2 = ax1.twinx()
cum_sum_df["nDCG"].plot(ax=ax2, color="red", legend=True)
ax2.legend(loc="lower right")
# ax2.set_ylabel("nDCG (Rescaled)", color="red")

_, cum_plot_col, _ = st.columns([.2, .3, .2])
with cum_plot_col:
    st.pyplot(fig)

st.dataframe(cum_sum_df.iloc[[50,0,1,4,9,24]])
