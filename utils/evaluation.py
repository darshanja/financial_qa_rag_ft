import time

def evaluate_models(questions, answers, rag_fn, ft_fn):
    results = []
    for q, a in zip(questions, answers):
        start = time.time()
        rag_answer = rag_fn(q)
        rag_time = time.time() - start

        start = time.time()
        ft_answer = ft_fn(q)
        ft_time = time.time() - start

        results.append({
            "question": q,
            "ground_truth": a,
            "rag_answer": rag_answer,
            "rag_time": round(rag_time, 2),
            "ft_answer": ft_answer,
            "ft_time": round(ft_time, 2)
        })
    return results
