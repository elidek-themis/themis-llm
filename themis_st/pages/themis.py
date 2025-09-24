import streamlit as st

from themis.data.repository import get_repo

st.title("Welcome to Themis ⚖️")

if "repo" not in st.session_state:
    with st.spinner("Loading repository..."):
        st.session_state.repo = get_repo()
        st.session_state.repo.sort_values(by="model", inplace=True)

no_experiments = len([data for data in st.session_state.repo.data if not data.is_group])
st.subheader(f"No experiments: {no_experiments}")

st.dataframe(st.session_state.repo.astype(str))

st.write(st.session_state.repo.iloc[0].data.is_group)


st.markdown(
    """
    ### Notes
    * Pick Q8_0 if it fits. There is practically no point to run fp16/bf16 versions since Q8 is nearly loseless with half of the memory usage.
    * Choose the largest K-quant that can fit.
    * If you are on a GPU and need to go below Q4, pick the biggest IQ quant. Those are slightly higher quality for the size, but slower for CPU inference.
    * If you need to go below Q3, consider running a smaller sized model.
    """
)
