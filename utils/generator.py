from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def generate_answer(query: str, context_chunks: List[str], max_length: int = 100) -> str:
    """
    Generate answer from context chunks using a language model.
    
    Args:
        query: User's question
        context_chunks: Retrieved context chunks
        max_length: Maximum answer length
        
    Returns:
        Generated answer
    """
    # Clean and combine chunks
    cleaned_chunks = []
    for chunk in context_chunks:
        # Remove any special characters and extra whitespace
        cleaned = ' '.join(chunk.split())
        if cleaned:
            cleaned_chunks.append(cleaned)
    
    context = ' '.join(cleaned_chunks)
    
    # Truncate context if too long (keeping last 1000 words as they're usually most relevant)
    context_words = context.split()
    if len(context_words) > 1000:
        context = ' '.join(context_words[-1000:])
    
    # Create focused prompt
    prompt = ("Based on the following financial report excerpt, "
             "provide a clear and concise answer to the question. "
             "If the information is not in the context, say 'I cannot find this information in the financial documents.'\n\n"
             f"Context: {context}\n\n"
             f"Question: {query}\n\n"
             "Answer:")
             
    # Initialize model and tokenizer
    model_name = "distilgpt2"  # or any other suitable model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
    
    # Generate answer
    response = generator(prompt, max_new_tokens=max_length, 
                       do_sample=True, temperature=0.7, 
                       num_return_sequences=1)[0]["generated_text"]
    
    # Extract answer from response
    answer = response.split("Answer:")[-1].strip()
    
    # Clean up the answer
    answer = answer.replace("<|endoftext|>", "").strip()
    
    return answer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
