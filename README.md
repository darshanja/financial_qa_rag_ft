# Financial QA System: RAG vs Fine-Tuning

This project implements and compares two approaches for answering questions about Allstate's financial reports:
1. **Retrieval-Augmented Generation (RAG)**: Combines hybrid document retrieval with generative language models
2. **Fine-Tuned Language Model (FT)**: Direct fine-tuning of a small language model on financial Q&A

## 🚀 Quick Start

### Model Files
Due to file size limitations, model files are not included in this repository. To use the system:

1. Download the fine-tuned model from [Hugging Face Hub](https://huggingface.co/models) (link to be added)
2. Place the downloaded files in the following structure:
   ```
   models/
   └── fine_tuned_model/
       ├── config.json
       ├── model.safetensors
       ├── tokenizer.json
       └── ...
   ```

## 🌟 Key Features

### RAG System
- **Hybrid Retrieval**: 
  - Dense retrieval using Sentence Transformers (all-MiniLM-L6-v2)
  - Sparse retrieval using BM25
  - Score fusion for optimal chunk selection
- **Context-Aware Generation**: 
  - Prompts engineered for financial accuracy
  - Dynamic context window management
  - Multi-chunk answer synthesis

### Fine-Tuned Model
- **Base Model**: DistilGPT2 (small, efficient)
- **Training Data**: 30+ carefully curated financial Q&A pairs
- **Optimization**: Parameter-efficient fine-tuning

### Guardrails
- **Input Validation**:
  - Financial keyword detection
  - Query complexity analysis
  - Minimum length requirements
- **Output Validation**:
  - Confidence scoring
  - Hallucination detection
  - Answer quality metrics

### Evaluation Framework
- Response time tracking
- Confidence scoring
- Chunk relevance metrics
- Answer quality assessment

## 🔧 Setup Instructions (Run Locally)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd financial_qa_rag_ft
```

### 2. Create Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Financial Reports
```bash
python utils/download_reports.py
```

### 5. Extract and Clean Text
```bash
Run: notebooks/01_data_preprocessing.ipynb
```

### 6. Generate QA Pairs Automatically
```bash
python utils/generate_qa_pairs.py
```

### 7. Train Fine-Tuned Model (Optional)
```bash
Run: notebooks/03_fine_tuning.ipynb
```

### 8. Run Streamlit App
```bash
streamlit run app/app.py
```

---

## 📂 Project Structure
```
financial_qa_rag_ft/
├── app/
│   └── app.py                 # Streamlit web interface with real-time metrics
├── data/
│   ├── processed/             # Cleaned and segmented text files
│   │   ├── Allstate_2022_10K.txt
│   │   └── Allstate_2023_10K.txt
│   └── raw/                   # Original financial reports
│       ├── Allstate_2022_10K.pdf
│       └── Allstate_2023_10K.pdf
├── models/
│   ├── fine_tuned_model/     # DistilGPT2 fine-tuned on financial QA
│   └── rag_model/            # Saved embeddings and retrieval indices
├── notebooks/
│   ├── 01_data_preprocessing.ipynb  # PDF parsing and text cleaning
│   ├── 02_rag_pipeline.ipynb       # RAG implementation and testing
│   ├── 03_fine_tuning.ipynb       # Model fine-tuning process
│   ├── 04_evaluation.ipynb        # Individual model evaluation
│   └── 05_evaluation_comparison.ipynb  # Comparative analysis
├── qa_pairs/
│   └── qa_dataset.json       # Curated financial QA pairs
├── utils/
│   ├── chunking.py           # Smart text segmentation
│   ├── data_preprocessing.py # PDF processing pipeline
│   ├── evaluation.py        # Comprehensive metrics
│   ├── fine_tuning.py      # Training utilities
│   ├── generator.py        # Answer generation logic
│   ├── guardrails.py      # Input/output validation
│   └── retriever.py       # Hybrid search implementation
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

---

## 📊 Performance Comparison

### RAG System
- **Strengths**:
  - Higher factual accuracy
  - Better source traceability
  - More robust to unseen questions
- **Metrics**:
  - Average response time: ~0.5s
  - Typical confidence: 0.8-0.95
  - Strong chunk relevance scores

### Fine-tuned Model
- **Strengths**:
  - Faster inference
  - More natural language
  - Consistent response style
- **Metrics**:
  - Average response time: ~0.4s
  - Typical confidence: 0.75-0.9
  - Good performance on seen patterns

## 💡 Example Questions

```python
# High-confidence questions
"What was Allstate's total revenue in 2023?"
"How much was the net loss in 2023?"
"What were the total assets in 2022?"

# Complex analytical questions
"How did revenue change from 2022 to 2023?"
"What factors affected profitability in 2023?"
"Compare the investment portfolio returns between 2022 and 2023"
```

## 🛠️ Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.31+
- Streamlit 1.24+
- Sentence-Transformers 2.2+
- See requirements.txt for full list

## � License
This project is for academic/educational use only. Financial data sourced from Allstate's public reports.

## 🙏 Acknowledgments
- Built using Hugging Face Transformers
- Financial data from Allstate's 10-K reports
- Streamlit for the web interface
