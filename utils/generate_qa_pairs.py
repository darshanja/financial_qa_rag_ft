import os
import json

def generate_qa_from_text(text, year):
    qa_pairs = []
    lines = text.splitlines()
    for line in lines:
        if "revenue" in line.lower():
            qa_pairs.append({
                "question": f"What was the company’s revenue in {year}?",
                "answer": line.strip()
            })
        if "net income" in line.lower():
            qa_pairs.append({
                "question": f"What was the net income in {year}?",
                "answer": line.strip()
            })
    return qa_pairs

if __name__ == "__main__":
    data_dir = "data/processed"
    output_path = "qa_pairs/qa_dataset.json"
    all_pairs = []

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            year = file.split("_")[1]
            with open(os.path.join(data_dir, file)) as f:
                text = f.read()
            qa = generate_qa_from_text(text, year)
            all_pairs.extend(qa)

    with open(output_path, "w") as out:
        json.dump(all_pairs, out, indent=2)
    print(f"Generated {len(all_pairs)} Q/A pairs.")
