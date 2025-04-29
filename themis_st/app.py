import streamlit as st

from st_pages import get_nav_from_toml

# st.set_page_config(
#     page_title="Themis",
#     page_icon="⚖️",
#     layout="wide"
# )

st.set_page_config(layout="wide")

nav = get_nav_from_toml(".streamlit/pages_sections.toml")

pg = st.navigation(nav)
pg.run()
