"""
Fine-tuning utilities for language models.
"""
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import torch
import os
from typing import List, Dict

class FinancialQATrainer:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        output_dir: str = "models/fine_tuned_model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer for financial QA fine-tuning.
        
        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
            device: Device to use for training
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = device
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Add special tokens if needed
        special_tokens = ["<|question|>", "<|answer|>", "<|endoftext|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def prepare_dataset(self, qa_pairs: List[Dict]) -> Dataset:
        """
        Prepare dataset for fine-tuning.
        
        Args:
            qa_pairs: List of question-answer pairs
            
        Returns:
            HuggingFace Dataset
        """
        # Format data
        formatted_data = []
        for pair in qa_pairs:
            text = f"<|question|>{pair['question']}<|answer|>{pair['answer']}<|endoftext|>"
            formatted_data.append({"text": text})
            
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=True
        )
        
        return tokenized_dataset
        
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        logging_steps: int = 100
    ):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Number of steps between logging
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
    @classmethod
    def from_qa_file(cls, qa_file_path: str, **kwargs):
        """
        Create trainer instance and prepare dataset from QA file.
        
        Args:
            qa_file_path: Path to JSON file containing QA pairs
            **kwargs: Additional arguments for trainer initialization
            
        Returns:
            Tuple of (trainer, train_dataset, eval_dataset)
        """
        # Load QA pairs
        with open(qa_file_path, 'r') as f:
            qa_pairs = json.load(f)
            
        # Create trainer
        trainer = cls(**kwargs)
        
        # Split data into train/eval
        train_size = int(0.8 * len(qa_pairs))
        train_pairs = qa_pairs[:train_size]
        eval_pairs = qa_pairs[train_size:]
        
        # Prepare datasets
        train_dataset = trainer.prepare_dataset(train_pairs)
        eval_dataset = trainer.prepare_dataset(eval_pairs)
        
        return trainer, train_dataset, eval_dataset
        
def fine_tune_model(
    qa_file_path: str,
    model_name: str = "distilgpt2",
    output_dir: str = "models/fine_tuned_model",
    **training_kwargs
):
    """
    Convenience function to fine-tune a model on QA pairs.
    
    Args:
        qa_file_path: Path to JSON file containing QA pairs
        model_name: Base model to fine-tune
        output_dir: Directory to save fine-tuned model
        **training_kwargs: Additional arguments for training
    """
    # Initialize trainer and prepare datasets
    trainer, train_dataset, eval_dataset = FinancialQATrainer.from_qa_file(
        qa_file_path,
        model_name=model_name,
        output_dir=output_dir
    )
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **training_kwargs
    )
    
if __name__ == "__main__":
    # Example usage
    qa_file = "qa_pairs/qa_dataset.json"
    fine_tune_model(
        qa_file,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )
