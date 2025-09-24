import sys

import streamlit as st

from st_pages import get_nav_from_toml

from themis.definitions.constants import ROOT_PATH  # noqa: F401

# pages_path = ROOT_PATH / "themis_st" / ".streamlit" / "pages_sections.toml"

# https://github.com/streamlit/streamlit/issues/10992
torch_mod = sys.modules.get("torch")
if torch_mod and hasattr(torch_mod, "classes"):
    torch_mod.classes.__path__ = []

st.set_page_config(layout="wide")

nav = get_nav_from_toml("themis_st/.streamlit/pages_sections.toml")

pg = st.navigation(nav)
pg.run()
