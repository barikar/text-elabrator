import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="Text Elaborator", layout="centered")

st.title("🔍 Text Elaborator using Hugging Face Transformers")

# User input
text = st.text_area("Enter a word, phrase, or short sentence to elaborate on:", height=100)

# Load text generator
if 'generator' not in st.session_state:
    st.session_state.generator = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )

if st.button("Elaborate Text"):
    if text.strip():
        with st.spinner("Generating elaboration..."):
            prompt = f"Explain in detail: {text.strip()}"
            output = st.session_state.generator(prompt, max_length=200, do_sample=True, temperature=0.7)
            st.success("Here's your detailed explanation:")
            st.write(output[0]['generated_text'].replace(prompt, "").strip())
    else:
        st.warning("Please enter a word or short phrase to elaborate.")