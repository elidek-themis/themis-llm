from dataclasses import dataclass

import pandas as pd
import streamlit as st

from themis_st.connections.hf_datasets import HFDatasetConnection


@dataclass
class SelectedDataset:
    """State object for the selected dataset"""

    name: str | None = None
    config: str | None = None
    split: str | None = None

    validity: dict[str, str] | None = None
    infos: dict[str, str] | None = None

    preview: pd.DataFrame | None = None

    @property
    def configs(self) -> list[str]:
        assert self.infos is not None, "Dataset infos are not loaded."
        return list(self.infos.keys())

    @property
    def config_args(self) -> dict:
        return {
            "options": self.configs,
            "index": self.configs.index(self.config) if self.config else None,
        }

    def _splits(self, config) -> list[str]:
        assert self.infos is not None, "Dataset infos are not loaded."
        return list(self.infos[config]["splits"])

    def splits_args(self, config) -> dict:
        splits = self._splits(config)

        return {
            "options": splits,
            "index": splits.index(self.split) if self.split else None,
        }

    @property
    def has_preview(self) -> bool:
        return self.validity.get("preview", False)

    @property
    def complete(self) -> bool:
        return all((self.name, self.config, self.split))


def save_dataset():
    st.session_state.dataset.config = None
    st.session_state.dataset.name = st.session_state._dataset

    r = st.session_state.hf_conn.is_valid(dataset=st.session_state.dataset.name)
    json = r.json()
    if r.ok:
        st.session_state.dataset.validity = json
    else:
        st.session_state.dataset.validity = None
        st.error(f"{r.status_code}: {json['error']}")


@st.cache_data
def get_infos(dataset: str) -> dict | None:
    r = st.session_state.hf_conn.get_info(dataset=dataset)

    json = r.json()
    if r.ok:
        return json["dataset_info"]
    else:
        st.error(f"{r.status_code}: {json['error']}")


def preview(selected_config: str, selected_split: str):
    st.session_state.dataset.config = selected_config
    st.session_state.dataset.split = selected_split

    r = st.session_state.hf_conn.preview(
        dataset=st.session_state.dataset.name,
        config=selected_config,
        split=selected_split,
    )

    json = r.json()
    if r.ok:
        st.session_state.dataset.preview = json
    else:
        st.session_state.dataset.preview = None
        st.error(f"{r.status_code}: {json['error']}")


@st.fragment
def config_fragment():
    st.session_state.dataset.infos = get_infos(st.session_state.dataset.name)
    selected_config = st.selectbox(
        label="Select Config",
        on_change=lambda: setattr(st.session_state.dataset, "split", None),
        **st.session_state.dataset.config_args,
    )
    if selected_config:
        selected_split = st.selectbox(label="Select Split", **st.session_state.dataset.splits_args(selected_config))
        if selected_split:
            click = st.button("Preview", on_click=preview, args=(selected_config, selected_split))
            if click:  # st.rerun within a callback is a no-op
                st.rerun()


if "dataset" not in st.session_state:
    st.session_state.dataset = SelectedDataset()

if "hf_conn" not in st.session_state:
    st.session_state.hf_conn = st.connection("hf", type=HFDatasetConnection)

with st.sidebar:
    with st.form("dataset_form"):
        st.text_input(label="🤗 Dataset Name", value=st.session_state.dataset.name, key="_dataset")
        st.form_submit_button("Check Validity", on_click=save_dataset)
    if st.session_state.dataset.validity:
        with st.expander("Validity", expanded=False):
            st.write(st.session_state.dataset.validity)
        if st.session_state.dataset.has_preview:
            config_fragment()

# GUARDS start
if st.session_state.dataset.validity is None:
    st.warning("Please enter a dataset name and check its validity.")
    st.stop()

if not st.session_state.dataset.validity["preview"]:
    st.error(f"Cannot preview dataset: {st.session_state.dataset.name}. Consider downloading it.")
    st.stop()
# GUARDS end

preview_col, info_col = st.columns([0.75, 0.25])

if st.session_state.dataset.complete:
    with info_col:
        infos = st.session_state.dataset.infos[st.session_state.dataset.config]
        st.json(infos, expanded=False)

    with preview_col:
        st.subheader("Dataset Preview")
        ds = st.session_state.dataset.preview
        st.write(pd.DataFrame([d["row"] for d in ds["rows"]]))
        # df = pd.DataFrame(st.session_state.dataset.preview["rows"])
        # st.dataframe(df, use_container_width=True)
