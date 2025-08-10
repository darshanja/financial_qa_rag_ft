"""
Input and output validation for the Financial QA system.
"""

def validate_input(query: str) -> tuple[bool, str]:
    """
    Validate user input query.
    
    Args:
        query: User's question
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check if query is empty
    if not query.strip():
        return False, "Please enter a question."
    
    # Check minimum length
    if len(query.split()) < 2:
        return False, "Please enter a complete question."
    
    # Check if query is finance-related
    financial_keywords = [
        'revenue', 'profit', 'loss', 'income', 'expense',
        'asset', 'liability', 'equity', 'cash', 'stock',
        'share', 'dividend', 'market', 'financial', 'fiscal',
        'quarter', 'annual', 'balance', 'statement', 'report',
        'earnings', 'cost', 'price', 'allstate', 'insurance'
    ]
    
    if not any(keyword.lower() in query.lower() for keyword in financial_keywords):
        return False, "Please ask a finance-related question about Allstate."
        
    return True, ""

def validate_output(answer: str, confidence: float) -> tuple[bool, str]:
    """
    Validate model output.
    
    Args:
        answer: Generated answer
        confidence: Model's confidence score
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check if answer is empty
    if not answer.strip():
        return False, "No answer generated."
    
    # Check if confidence is too low
    if confidence < 0.3:
        return False, "Low confidence in answer. Please rephrase your question."
    
    # Check for potential hallucination markers
    hallucination_markers = [
        "I don't know",
        "I'm not sure",
        "I cannot",
        "I don't have",
        "not available",
        "no information"
    ]
    
    if any(marker.lower() in answer.lower() for marker in hallucination_markers):
        return False, "Unable to find relevant information in the financial documents."
        
    return True, ""
