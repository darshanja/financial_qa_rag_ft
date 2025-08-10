# Models Directory

This directory is a placeholder for models, but the application now uses Hugging Face to host and load models.

## Current Implementation:

1. **Hugging Face Model**: The fine-tuned model is hosted at [jayyd/financial-qa-model](https://huggingface.co/jayyd/financial-qa-model) on Hugging Face Hub
   - The model was generated from the fine-tuning process in `notebooks/03_fine_tuning.ipynb`
   - The application in `app/app.py` loads the model directly from Hugging Face

## How to Generate and Push Models:

To generate and upload the fine-tuned model:
1. Run the notebook `notebooks/03_fine_tuning.ipynb` which will:
   - Fine-tune the model
   - Save it locally to this directory
   - Push it to Hugging Face Hub
   - Update the application to use the Hugging Face model

## Local Model Usage (Optional):

If you want to use a local model instead:
1. Download the model from Hugging Face using:
   ```
   python -c "from huggingface_hub import snapshot_download; snapshot_download('jayyd/financial-qa-model', local_dir='fine_tuned_model')"
   ```
2. Update `app/app.py` to use the local model:
   ```python
   model = AutoModelForCausalLM.from_pretrained("models/fine_tuned_model")
   tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned_model")
   ```
