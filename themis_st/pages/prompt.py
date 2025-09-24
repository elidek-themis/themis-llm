import math
import time

from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from annotated_text import annotation, annotated_text

from themis_st.processing.utils import format_jinja
from themis_st.connections.vllm_connection import VLLMConnection

st.title("Prompt Explorer")

if "conn" not in st.session_state:
    st.session_state.conn = None

if "completion" not in st.session_state:
    st.session_state.completion = None

if "credentials" not in st.session_state:
    st.session_state.url = None
    st.session_state.api_key = None
    st.session_state.credentials = {}


@st.fragment(run_every=30)
def is_healthy() -> bool:
    timestamp = time.time()
    timestamp = datetime.fromtimestamp(timestamp)
    timestamp = timestamp.strftime("%d/%m/%Y - %H:%M:%S")

    try:
        r = st.session_state.conn.health()
        if r.status_code == 200:  # noqa: PLR2004
            st.write(f"🟢 {timestamp}")
            return True
        else:
            st.write(f"🔴 {timestamp}")
            return False
    except requests.exceptions.ConnectionError:
        st.write(f"🔴 {timestamp}")
        return False


def get_models() -> list[str]:
    r = st.session_state.conn.get_models().json()
    models = [d["id"] for d in r["data"]]
    return models


def assign_model() -> None:
    st.session_state.selected_model = st.session_state._selected_model

    st.session_state.conn.assign_model(st.session_state.selected_model)
    st.toast(f"Assigned model: {st.session_state.selected_model}")
    time.sleep(0.5)  # interactivity hack


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    st.session_state.conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    st.session_state.credentials = credentials


with st.sidebar:
    with st.form("connection_form"):
        url = st.text_input(
            label="url", value=st.session_state.credentials.get("base_url", None), placeholder="http://localhost:8000"
        )
        api_key = st.text_input(
            label="api_key", value=st.session_state.credentials.get("token", None), placeholder="EMPTY", type="password"
        )

        if st.form_submit_button("Connect"):
            connect(url, api_key)

    if st.session_state.get("conn", None):
        if is_healthy():
            models = get_models()
            index = (
                models.index(st.session_state.selected_model) if st.session_state.get("selected_model", False) else None
            )

            st.selectbox(
                label="Select a model",
                options=models,
                index=index,
                on_change=assign_model,
                key="_selected_model",
                help="Selecting a model allows to access its tokenizer and chat template.",
            )


def submit() -> None:
    if not st.session_state.get("selected_model", False):
        st.error("No selected model")

    if st.session_state.prompt:
        st.session_state.completion = st.session_state.conn.completions.create(
            model=st.session_state.selected_model,
            prompt=st.session_state.prompt,
            logprobs=100,
            max_tokens=1,
            temperature=0,
            extra_body={"prompt_logprobs": 100},
        )
    else:
        st.session_state.completion = None
        st.toast("Oops")
        time.sleep(0.75)


def get_hex_color(value: float) -> str:
    rgba = plt.get_cmap("RdYlGn")(value)
    return mcolors.rgb2hex(rgba)  # Convert to HEX (e.g., '#a6d96a')


def tokenize():
    chat_history = []

    if st.session_state.system:
        chat_history.append({"role": "system", "content": st.session_state.system})

    if st.session_state.user:
        chat_history.append({"role": "user", "content": st.session_state.user})

    if st.session_state.assistant:
        chat_history.append({"role": "assistant", "content": st.session_state.assistant})

    tok = st.session_state.conn.model.apply_chat_template(
        chat_history=chat_history, add_generation_prompt=st.session_state.add_gen
    )

    st.session_state.prompt = tok
    submit()


if st.session_state.conn and st.session_state.get("selected_model", False):
    base_tab, chat_tab = st.tabs(["Base", "Instruct"])

    with base_tab:
        with st.form("base_form"):
            st.text_input(label="#### Prompt", key="prompt")
            st.form_submit_button("Generate", on_click=submit)

    with chat_tab:
        if st.session_state.conn.model.tokenizer.chat_template is None:
            st.error(f"{st.session_state.selected_model} has no chat template.")
        else:
            text_col, template_col = st.columns((0.5, 0.5))
            with text_col:
                with st.form("chat_form"):
                    st.text_area(label="#### System", height=68, key="system")
                    st.text_area(label="#### User", height=150, key="user")
                    st.text_area(label="#### Assistant", height=68, key="assistant")
                    st.checkbox("add_generation_prompt", key="add_gen")
                    st.form_submit_button("Tokenize", on_click=tokenize)
            with template_col:
                template = st.session_state.conn.model.tokenizer.chat_template
                fmt_template = format_jinja(template)
                st.code(fmt_template.strip(), height=500, language="jinja2")

    if st.session_state.completion:
        (next_token,) = st.session_state.completion.choices
        (next_logprobs,) = next_token.logprobs.top_logprobs
        prompt_logprobs = next_token.prompt_logprobs

        encoded_prompt = []
        for prompt in prompt_logprobs[1:]:
            token_id = next(iter(prompt))
            encoded_prompt.append({"token": token_id} | prompt[token_id])

        encoded_prompt_df = pd.DataFrame(encoded_prompt)
        encoded_prompt_df["probability"] = encoded_prompt_df.logprob.apply(lambda x: math.exp(x))

        annotations = []
        for _, row in encoded_prompt_df.iterrows():
            token = row.decoded_token
            prob = row.probability
            color = get_hex_color(prob)
            annotations.append(annotation(token, color="black", border=f"; background: {color}"))
        annotated_text(*annotations)

        # encoded_prompt = encoded_prompt.set_index("decoded_token").T

        next_tab, prompt_tab = st.tabs(["next token", "prompt"])
        with next_tab:
            next_logprobs = pd.Series(next_logprobs)
            next_logprobs = next_logprobs.to_frame("logprob").reset_index()
            next_logprobs["probability"] = next_logprobs.logprob.apply(lambda x: math.exp(x))
            st.dataframe(next_logprobs)
        with prompt_tab:
            st.dataframe(encoded_prompt)
#     annotated_text(
#         annotation("world!", color="white", border="1px line red"),
#         annotation("world!", color="#8ef", border="1px red"),
#     )

# st.selectbox(
#     label="Select token",
#     options=["Green", "Yellow", "Red", "Blue"],
# )
