# Filename: train_ner_final_jsonl.py
import os
from datasets import load_dataset, Dataset # We now need load_dataset
# No longer need to import from your local ner_data file
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

# --- Configuration ---
BASE_MODEL = "distilbert-base-uncased"
TRAIN_FILE = "../../data/ner_train.jsonl" # The script will now use this file
OUTPUT_DIR = "./models/ner-extractor"

print("--- Starting Final NER Model Training from JSONL ---")

# --- Step 1: Load the Dataset Directly from JSONL ---
# This single line replaces the entire manual pre-processing loop.
# It expects each line in the JSONL to have "tokens": [...] and "labels": [...]
print(f"Loading dataset from {TRAIN_FILE}...")
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
print(f"✅ Dataset loaded with {len(dataset)} examples.")


# --- Step 2: Infer Labels Correctly ---
# This logic works perfectly on the Dataset object loaded from the file.
print("Inferring labels from the dataset...")
uniq = set()
for row in dataset:
    uniq.update(row['labels'])
labels_list = sorted(list(uniq))
id2label = {i: l for i, l in enumerate(labels_list)}
label2id = {l: i for i, l in enumerate(labels_list)}
print(f"✅ Labels inferred: {labels_list}")


# --- Step 3: Initialize Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


# --- Step 4: Define Tokenization and Label Alignment ---
# This function remains unchanged and is correct.
def tokenize_and_align_labels(example):
    tokenized_input = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length"
    )
    word_ids = tokenized_input.word_ids()
    
    word_label_ids = [label2id[l] for l in example["labels"]]
    
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(word_label_ids[word_idx])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx
        
    tokenized_input["labels"] = aligned_labels
    return tokenized_input

print("Tokenizing and aligning labels...")
tokenized_dataset = dataset.map(tokenize_and_align_labels, remove_columns=dataset.column_names)
print("✅ Tokenization complete.")


# --- Steps 5, 6, 7, 8, 9 (No Changes) ---
# The rest of the script is identical as it correctly operates on the tokenized_dataset.

# Step 5: Load Model
model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL, num_labels=len(labels_list), id2label=id2label, label2id=label2id
)

# Step 6: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./ner_model_results",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    logging_steps=10,
)

# Step 7: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Step 8: Train
print("--- Starting Training ---")
trainer.train()
print("✅ Training complete.")

# Step 9: Save Final Model
print(f"Saving model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- Model saved successfully. ---")