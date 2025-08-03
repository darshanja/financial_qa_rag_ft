import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils.chunking import chunk_text
from utils.retriever import HybridRetriever
from utils.generator import generate_answer
import os
import json

st.set_page_config(page_title="Allstate Financial QA")

st.title("📊 Allstate Financial QA System")

query = st.text_input("Ask a financial question")
method = st.radio("Choose method:", ["RAG", "Fine-Tuned"])

if query:
    if method == "RAG":
        # Prepare data and retriever
        if "retriever" not in st.session_state:
            texts = []
            for file in os.listdir("data/processed"):
                with open(os.path.join("data/processed", file), "r") as f:
                    texts.append(f.read())
            chunks = chunk_text(texts, chunk_size=100)
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            retriever = HybridRetriever(chunks, embedder)
            st.session_state.retriever = retriever
        else:
            retriever = st.session_state.retriever

        chunks = retriever.retrieve(query)
        answer = generate_answer(query, chunks)
        st.success("Answer:")
        st.write(answer)

    else:
        # Load fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("models/fine_tuned_model")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt = f"Q: {query}\nA:"
        output = pipe(prompt, max_new_tokens=50)[0]['generated_text']
        st.success("Answer:")
        st.write(output)
