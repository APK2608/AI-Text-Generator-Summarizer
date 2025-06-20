import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Text AI App", layout="centered")
st.title("🧠 APK AI Text Generator & Summarizer") 

mode = st.selectbox("Choose Mode", ["Text Generation", "Summarization"])

if mode == "Text Generation":
    model = pipeline("text-generation", model="gpt2")
    prompt = st.text_input("Enter your prompt:")
    max_length = st.slider("Max length", 50, 500, 100)

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating..."):
                result = model(prompt, max_length=max_length, num_return_sequences=1)
                st.success("Generated Text:")
                st.write(result[0]["generated_text"])
        else:
            st.warning("Please enter a prompt.")

elif mode == "Summarization":
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    text = st.text_area("Paste your content to summarize:", height=500)

    if st.button("Summarize"):
        if text:
            with st.spinner("Summarizing..."):
                result = model(text, max_length=120, min_length=30, do_sample=False)
                st.success("Summary:")
                st.write(result[0]["summary_text"])
        else:
            st.warning("Please paste some text.") 
