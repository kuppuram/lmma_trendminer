# Filename: convert_ner_data_to_jsonl.py
import json
import json, random
from typing import List, Dict
# from ner_data import get_ner_data
from sklearn.model_selection import train_test_split
import datasets.ner_data

# Get the data from your original file

# def _load_py(module_name: str) -> List[Dict]:
#     mod = __import__(module_name, fromlist=["get_ner_data"])
#     data = mod.get_ner_data()
#     if not isinstance(data, list):
#         raise ValueError("Expected get_ner_data() -> List[dict] with keys 'text' and 'labels'")
#     return data

all_data = datasets.ner_data.get_ner_data()

# Reformat the data into the required structure
formatted_data = []
for item in all_data:
    # The ner_cli.py script expects a list of tokens, not a single string
    tokens = item['text'].split()
    labels = item['labels'].split()
    
    if len(tokens) == len(labels):
        formatted_data.append({"tokens": tokens, "labels": labels})

# Split the data into training and validation sets
train_data, eval_data = train_test_split(formatted_data, test_size=0.2, random_state=42)

# Write to JSONL files
def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

write_jsonl(train_data, 'ner_train.jsonl')
write_jsonl(eval_data, 'ner_eval.jsonl')

print("✅ Successfully created train.jsonl and eval.jsonl")