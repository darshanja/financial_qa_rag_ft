import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
import sys
import os
import json

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.chunking import smart_chunk_text
from utils.retriever import HybridRetriever
from utils.generator import generate_answer
from utils.evaluation import evaluate_response
from utils.guardrails import validate_input, validate_output


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Allstate Financial QA")
st.title("📊 Allstate Financial QA System")


# ---------------------------
# Cached Loaders
# ---------------------------
@st.cache_resource
def load_retriever():
    texts = []
    for file in os.listdir("data/processed"):
        if file.endswith(".txt") or file.endswith(".json"):
            with open(os.path.join("data/processed", file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    chunks = smart_chunk_text(texts, chunk_size=100)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return HybridRetriever(chunks, embedder)


@st.cache_resource
def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained("jayyd/financial-qa-model")
    model = AutoModelForCausalLM.from_pretrained("jayyd/financial-qa-model")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


# ---------------------------
# UI Inputs
# ---------------------------
query = st.text_input("Ask a financial question")
method = st.radio("Choose method:", ["RAG", "Fine-Tuned"])


# ---------------------------
# Main App Logic
# ---------------------------
if query:
    # Validate input
    is_valid, message = validate_input(query)
    if not is_valid:
        st.error(message)
        st.stop()

    start_time = time.time()

    if method == "RAG":
        with st.spinner("Retrieving and generating answer..."):
            retriever = load_retriever()
            chunks = retriever.retrieve(query)
            # Extractive only
            answer, supporting_context = generate_answer(query, chunks)

            # Evaluate
            metrics = evaluate_response(query, answer, chunks)
            confidence = metrics.get("confidence", 0.0)
            is_valid, message = validate_output(answer, confidence)

        # Show answer
        st.subheader("Answer")
        st.write(answer)

        if not is_valid:
            st.warning(message)

        # Supporting context (expandable)
        with st.expander("Supporting Context"):
            st.write(supporting_context)

        # Sidebar metrics
        response_time = time.time() - start_time
        st.sidebar.markdown("### Response Metrics")
        st.sidebar.markdown(f"Response Time: {response_time:.2f}s")
        st.sidebar.markdown(f"Confidence Score: {confidence:.2f}")
        st.sidebar.markdown(f"Number of Retrieved Chunks: {metrics.get('num_chunks', 0)}")
        st.sidebar.markdown(f"Chunk Relevance Score: {metrics.get('chunk_relevance', 0):.2f}")

    else:
        with st.spinner("Generating answer from fine-tuned model..."):
            pipe = load_finetuned_model()
            prompt = f"Q: {query}\nA:"
            raw_output = pipe(prompt, max_new_tokens=100)[0]["generated_text"]

            # Clean the output: remove prompt part
            output = raw_output.split("A:")[-1].strip()

            # Evaluate
            metrics = evaluate_response(query, output)
            confidence = metrics.get("confidence", 0.0)
            is_valid, message = validate_output(output, confidence)

        # Show answer
        st.subheader("Answer")
        st.write(output)

        if not is_valid:
            st.warning(message)

        # Sidebar metrics
        response_time = time.time() - start_time
        st.sidebar.markdown("### Response Metrics")
        st.sidebar.markdown(f"Response Time: {response_time:.2f}s")
        st.sidebar.markdown(f"Confidence Score: {confidence:.2f}")
        st.sidebar.markdown(f"Answer Length: {metrics.get('answer_length', 0)} words")