import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.chunking import chunk_text
from utils.retriever import HybridRetriever
from utils.generator import generate_answer
from utils.evaluation import calculate_confidence, evaluate_response
from utils.guardrails import validate_input, validate_output
import os
import json

st.set_page_config(page_title="Allstate Financial QA")

st.title("📊 Allstate Financial QA System")

query = st.text_input("Ask a financial question")
method = st.radio("Choose method:", ["RAG", "Fine-Tuned"])

if query:
    # Validate input
    is_valid, message = validate_input(query)
    if not is_valid:
        st.error(message)
        st.stop()

    start_time = time.time()
    
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
        
        # Evaluate response
        metrics = evaluate_response(query, answer, chunks)
        confidence = metrics["confidence"]
        
        # Validate output
        is_valid, message = validate_output(answer, confidence)
        if not is_valid:
            st.warning(message)
        else:
            st.success("Answer:")
            st.write(answer)
            
        # Display metrics
        response_time = time.time() - start_time
        st.sidebar.markdown("### Response Metrics")
        st.sidebar.markdown(f"Response Time: {response_time:.2f}s")
        st.sidebar.markdown(f"Confidence Score: {confidence:.2f}")
        st.sidebar.markdown(f"Number of Retrieved Chunks: {metrics['num_chunks']}")
        st.sidebar.markdown(f"Chunk Relevance Score: {metrics['chunk_relevance']:.2f}")

    else:
        # Load fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("models/fine_tuned_model")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt = f"Q: {query}\nA:"
        output = pipe(prompt, max_new_tokens=50)[0]['generated_text']
        
        # Evaluate response
        metrics = evaluate_response(query, output)
        confidence = metrics["confidence"]
        
        # Validate output
        is_valid, message = validate_output(output, confidence)
        if not is_valid:
            st.warning(message)
        else:
            st.success("Answer:")
            st.write(output)
            
        # Display metrics
        response_time = time.time() - start_time
        st.sidebar.markdown("### Response Metrics")
        st.sidebar.markdown(f"Response Time: {response_time:.2f}s")
        st.sidebar.markdown(f"Confidence Score: {confidence:.2f}")
        st.sidebar.markdown(f"Answer Length: {metrics['answer_length']} words")
