import json
import logging
import subprocess
import time
from ast import literal_eval

import pandas as pd
import requests
import streamlit as st
from code_editor import code_editor
from pydantic import ValidationError
from streamlit.logger import get_logger
from streamlit_extras.capture import logcapture
from vllm.engine.arg_utils import EngineArgs

from themis_st.processing.model import Connection
from themis_st.processing.utils import VLLMArgs, get_model_hub

# st.set_page_config(layout="wide")

TEMPLATE_CFG = {
    # "model": "facebook/opt-125m",
    "task": "generate",
    "max_model_len": 128,
    "max_logprobs": 20,
    "swap_space": 4,
    "cpu_offload_gb": 0,
    "gpu_memory_utilization": 0.9,
    "seed": 2025,
}

MODELS = get_model_hub()["model"].to_list()
MODELS = ["facebook/opt-125m"] if not MODELS else MODELS

st_logger = get_logger(__name__)


if "vllm_config" not in st.session_state:
    st.session_state.vllm_config = dict()

# if "credentials" not in st.session_state:
#     st.session_state.credentials = dict()

if "con" not in st.session_state:
    st.session_state.con = None


def validate_config(config):
    msg = st.toast("Validating config")
    time.sleep(0.5)
    if not config:
        st.toast("Save 💾")
        return
    try:
        config = literal_eval(config)
        msg.toast("EngineArgs validations..")
        EngineArgs(**{"model": st.session_state.selected_model, **config})
        time.sleep(0.5)
        msg.toast("EngineArgs validations ✔️")
        time.sleep(0.5)
        msg.toast("Pydantic validations ..")
        VLLMArgs(**{"model": st.session_state.selected_model, **config})
        time.sleep(0.5)
        msg.toast("Pydantic validations ✔️")
        time.sleep(0.5)
        msg.toast("Valid config ✔️")
        st.session_state.vllm_config = config
    except SyntaxError as e:
        st.toast(f":red[{e.msg} line{e.lineno}]")
    except ValidationError as e:
        st.toast(e)
    except TypeError as e:
        st.toast(e)
    finally:
        time.sleep(0.5)


def nvidia_smi():
    gpu_usage = subprocess.getoutput("nvidia-smi")
    return gpu_usage
    # gpu_usage = gpu_usage.split("\n")
    # return "\n".join(gpu_usage[7:12])


# . # # #
st.subheader(
    "OpenAI Compatible Server [🔗](https://https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)"
)
_, input_col, req_col, _ = st.columns([0.2, 0.3, 0.3, 0.2])
# . _ # _
_, log_col, _ = st.columns([0.2, 0.52, 0.2])
with log_col:
    log_widget = st.empty()

with input_col:
    st.write(
        "###### vLLM Engine Arguments [🔗](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#cli-reference)"
    )
    selected_model = st.session_state.get("selected_model", "facebook/opt-125m")
    # with st.form("from"):
    st.selectbox(label="HuggingFace Hub", options=MODELS, index=MODELS.index(selected_model), key="selected_model")

    code_tab, upload_tab = st.tabs(["Input", "Upload"])
    with code_tab:
        custom_btns = [
            {
                "name": "Save",
                "feather": "Save",
                "hasText": True,
                "alwaysOn": True,
                "commands": ["submit"],
                "style": {"bottom": "0.5rem", "right": "0.5rem"},
            }
        ]

        config = st.session_state.vllm_config
        config = config if config else TEMPLATE_CFG
        fmt_config = json.dumps(config, indent=4)
        code_dict = code_editor(
            fmt_config,
            lang="python",
            buttons=custom_btns,
            key="editor",
        )
    st.button("Validate", on_click=validate_config, kwargs={"config": code_dict["text"]})
    with upload_tab:
        uploaded_file = st.file_uploader("Choose config")


def connect():
    url = st.session_state.url
    url_hp = url + "/health"

    api_key = st.session_state.api_key
    msg = st.toast("Connecting to OpenAI compatible server ..")
    time.sleep(0.75)

    if url and api_key:
        try:
            response = Connection.request(method="GET", url=url_hp)
            msg.toast(f"{url_hp}: {response.status_code}")
            time.sleep(0.75)
            response.raise_for_status()
            st.session_state.con = Connection(url, api_key)
            msg.toast("Connecting to OpenAI compatible server ✔️")
        except requests.exceptions.RequestException:
            st.toast(f"Server {url} not reachable")
    else:
        st.toast("Oops")  # ("Please enter a valid URL and API key")


with req_col:
    st.markdown(
        "###### Connect to OpenAI Compatible Server [🔗](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#api-reference)"
    )
    with st.form("form"):
        url = st.session_state.get("url", None)
        api_key = st.session_state.get("api_key", None)

        st.text_input(label="url", value=url, placeholder="http://localhost:8080", key="url")
        st.text_input(label="api_key", value=api_key, placeholder="EMPTY", key="api_key")

        st.form_submit_button("Connect", on_click=connect)


# log_widget.code(nvidia_smi(), language="bash")
with st.sidebar:
    if st.button("nvidia-smi"):
        log_widget.code(nvidia_smi(), language="bash")
        # with logcapture(log_widget.code, from_logger=logger):
        #     st_logger.info()
    with st.expander("vLLM Config", expanded=True):
        config = {"model": st.session_state.selected_model}
        if st.session_state.vllm_config:
            config.update(**st.session_state.vllm_config)
            st.write(config)
            host_c, port_c = st.columns(2)
            host = host_c.text_input("Host", placeholder="localhost")
            port = port_c.text_input("Port", placeholder="8080")
            if st.button("Run server"):
                st.balloons()
        else:
            st.write("Config not set")
    with st.expander("Connection", expanded=True):
        if st.session_state.con:
            st.write(st.session_state.con.credentials)
            if st.button("Ping 〽️"):
                url = st.session_state.con.credentials["url"]
                response = Connection.request(method="GET", url=url + "/health")
                if response.status_code == 200:
                    st.toast("Pong 🏓")
        else:
            st.write("No connection")


# c = st.empty()
# with c.container():
#     for seconds in range(100):
#         c.write(f"⏳ {seconds} seconds have passed")
#         time.sleep(1)
#     c.write(":material/check: 10 seconds over!")
