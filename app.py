import streamlit as st
from generateScientificResponse import generateResponse
import json

st.set_page_config(page_title="Scientific AI Assistant", layout="wide")
st.title("🔬 Scientific Research Assistant")
st.markdown("Clear doubts related to LLMs and NLP based on real citations from small ArXiv database.")


with st.sidebar:
    st.header("System Status")
    st.success("Connected to Pinecone Index")
    st.info("Model: gpt-4o-mini + DeBERTa-v3")

query = st.text_input("Enter your scientific question:", placeholder="e.g. How do LLMs work?")

if query:
    with st.spinner("Searching papers and verifying claims..."):
        context, response = generateResponse(query)

        st.subheader("Answer")
        st.write(response)

        with st.expander("View Cited Sources & Evidence"):
            st.text(json.dumps(context))