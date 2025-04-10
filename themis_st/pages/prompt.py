import math
import time
import pandas as pd
import streamlit as st

from themis_st.processing.model import Connection
from annotated_text import annotated_text, annotation

st.title("Prompt Explorer")

if "con" not in st.session_state:
    st.session_state.con = None
    
if "completion" not in st.session_state:
    st.session_state.completion = None
    
with st.sidebar:
    with st.expander("Connection", expanded=True):
        if st.session_state.con:
            st.write(st.session_state.con.credentials)
            if st.button("Ping 〽️"):
                url = st.session_state.con.credentials["url"]
                response = Connection.request(method="GET", url=url+"/health")
                if response.status_code == 200:
                    st.toast("Pong 🏓")
        else:
            st.write("No connection")


def submit():
    if st.session_state.prompt:
        con = st.session_state.con
        st.session_state.completion = con.client.completions.create(
            model="meta-llama/Llama-3.1-8B",
            prompt=st.session_state.prompt,
            logprobs=20,
            max_tokens=1,
            temperature=0,
            extra_body={"prompt_logprobs": 20}
        )
    else:
        st.dialog("Oops")
        time.sleep(0.75)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_hex_color(value):
    rgba = plt.cm.RdYlGn(value)  # Get RGBA
    return mcolors.rgb2hex(rgba)  # Convert to HEX (e.g., '#a6d96a')

    
if st.session_state.con:
    with st.form("form"):
        st.text_input("#### Prompt", key="prompt")
        st.form_submit_button("Generate", on_click=submit)
    

    if st.session_state.completion:
        
        (next_token,) = st.session_state.completion.choices
        (next_logprobs,) = next_token.logprobs.top_logprobs
        prompt_logprobs = next_token.prompt_logprobs
        
        encoded_prompt = []
        for prompt in prompt_logprobs[1:]:
            token_id = next(iter(prompt))
            encoded_prompt.append({"token": token_id} | prompt[token_id])
        encoded_prompt = pd.DataFrame(encoded_prompt)
        encoded_prompt["probability"] = encoded_prompt.logprob.apply(lambda x: math.exp(x))
        
        annotations = []
        for i, row in encoded_prompt.iterrows():
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