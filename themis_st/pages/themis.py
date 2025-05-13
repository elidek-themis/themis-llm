import streamlit as st

from themis.data.repository import Repository
from themis_st.processing.utils import get_runs_df

st.title("Welcome to Themis ⚖️")

if "repo" not in st.session_state:
    with st.spinner("Loading repository..."):
        st.session_state.repo = Repository()
        runs = st.session_state.repo.db.table("experiments").all()
        st.session_state.runs = get_runs_df(runs)

if "run" not in st.session_state:
    st.session_state.run = None


def update_run() -> None:
    select_idx, *_ = st.session_state.runs_table.selection.rows
    output = st.session_state.runs.iloc[select_idx]["output"]
    st.session_state.run = output


model_tab, metric_tab, runs_tab = st.tabs(["models", "metrics", "experiments"])

with model_tab:
    st.write(st.session_state.repo.db.table("model_hub").all())
with metric_tab:
    st.write(st.session_state.repo.db.table("metrics").all())
with runs_tab:
    event = st.dataframe(
        st.session_state.runs,
        # use_container_width=True,
        hide_index=False,
        on_select=update_run,
        selection_mode="single-row",
        key="runs_table",
    )

    if st.session_state.run:
        st.write(st.session_state.run)
