import os.path as osp

import pandas as pd
import streamlit as st

from themis_st.processing.plot import catplot
from themis_st.processing.style import stats_styler
from themis.definitions.constants import RAW_PATH
from themis_st.processing.process import (
    ElectionResults,
    get_nll_df,
    get_voting,
    get_prob_df,
    get_differences,
    get_voting_stats,
)

voting_20 = get_voting(voting_path=osp.join(RAW_PATH, "voting-2020.xlsx"))
voting_24 = get_voting(voting_path=osp.join(RAW_PATH, "voting-2024.xlsx"))
runs = st.session_state.runs

model = st.selectbox(
    "Select model",
    runs.model.unique(),
)
model_runs = runs[runs.model == model]

tasks = st.multiselect("Select tasks", model_runs.task.sort_values())

for task in tasks:
    # task_name = task.replace("_", " ").title()
    # st.divider()
    with st.expander(f"#### {task}"):
        run = runs[(runs.model == model) & (runs.task == task)]
        # run = repo.load(model=model, task=task)
        output = run.output.item()

        task_config = output.task_configs[task]
        dataset_kwargs = task_config.get("dataset_kwargs")
        template = dataset_kwargs.get("template")
        # sub = dataset_kwargs.get("sub")
        # us_template = re.sub(string=template, **sub)
        # input_col.write({"template": template, "sub": sub, "U.S. template": us_template})

        st.markdown(
            f"""
        <style>
        .custom-font {{
            font-family: 'Courier New', monospace;
            font-size: 15px !important;
            color: orange;
        }}
        </style>
        <p class="custom-font">{template}</p>
        """,
            unsafe_allow_html=True,
        )

        choices = dataset_kwargs.get("choices")

        no_choices = int(len(choices) / 2)
        D_choices, R_choices = choices[:no_choices], choices[no_choices:]
        columns = dataset_kwargs.get("columns")

        menu_df = pd.DataFrame(
            {
                "Democratic": D_choices,
                "Republican": R_choices,
                "column": columns,
            }
        )

        with st.popover("Menu", use_container_width=True):
            st.dataframe(
                data=menu_df,
                hide_index=True,
            )

        results = ElectionResults(output=output)
        data = results.metrics[task]["acc"]
        nll_df = get_nll_df(
            data=data, index=results.keys, columns=results.choices[task], use_cols=results.choices[task]
        )
        prob_df, norm_prob_df = get_prob_df(nll_df=nll_df)
        diff = get_differences(df=norm_prob_df, columns=results.columns[task])
        metrics_col, catplot_col = st.columns([0.8, 0.4])
        with metrics_col:
            tab20, tab24 = st.tabs(["2020", "2024"])
            with tab20:
                stats = get_voting_stats(
                    voting=voting_20, norm_prob_df=norm_prob_df, diff=diff, columns=results.columns[task]
                )
                st.dataframe(stats_styler(stats))
            with tab24:
                stats = get_voting_stats(
                    voting=voting_24, norm_prob_df=norm_prob_df, diff=diff, columns=results.columns[task]
                )
                st.dataframe(stats_styler(stats))
        with catplot_col:
            cat_diff = diff.drop(["std", "se"], axis=1)
            cat_diff["pct_diff_20"] = voting_20.pct_diff
            cat_diff["pct_diff_24"] = voting_24.pct_diff

            box_col, violin_col = st.tabs([":package:", ":violin:"])
            with box_col:
                g = catplot(df=cat_diff, title="", kind="box")
                st.pyplot(g)
            with violin_col:
                g = catplot(df=cat_diff, title="", kind="violin")
                st.pyplot(g)

st.subheader("Aggregation")
st.write("TODO")
