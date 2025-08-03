# Financial QA System: RAG vs Fine-Tuning

This project demonstrates a comparative financial QA system built using:
- Retrieval-Augmented Generation (RAG)
- Fine-Tuned Language Model (FT)

Both systems answer questions from Allstate's 2022–2023 financial reports.

---

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
├── data/              # Raw and processed financial reports
├── qa_pairs/          # Generated QA pairs
├── models/            # Fine-tuned model and embeddings
├── utils/             # Chunking, retriever, generator, QA scripts
├── notebooks/         # Jupyter notebooks (preprocess, RAG, FT, eval)
├── app/               # Streamlit frontend
├── requirements.txt
├── README.md
└── Financial_QA_Report.pdf
```

---

## ✅ Features

- Hybrid retrieval (BM25 + embeddings)
- GPT2-based generation (RAG & FT)
- Streamlit UI with method toggle
- Input/output guardrails
- Screenshots and performance report

---

## 📄 License
This project is for academic/educational use only.
