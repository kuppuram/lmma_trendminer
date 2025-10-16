# Filename: train_ner_final.py
import os
from datasets import Dataset
import ner_datasets.ner_data as ner_data # Corrected import
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

# --- Configuration ---
BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = "./models/ner-extractor"

print("--- Starting Final NER Model Training ---")

# --- Step 1: Load and Pre-process the Raw Data ---
print("Loading and formatting raw data...")
raw_data = ner_data.get_ner_data()
formatted_data = []
for item in raw_data:
    tokens = item['text'].split()
    labels = item['labels'].split()
    if len(tokens) == len(labels):
        formatted_data.append({"tokens": tokens, "labels": labels})

dataset = Dataset.from_list(formatted_data)
print(f"✅ Dataset loaded and formatted with {len(dataset)} examples.")

# --- Step 2: Infer Labels Correctly ---
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
def tokenize_and_align_labels(example):
    # THIS IS THE CORRECTED TOKENIZER CALL
    tokenized_input = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length"  # This ensures all sequences have the same length
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

# --- Step 5: Load Model ---
model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL, num_labels=len(labels_list), id2label=id2label, label2id=label2id
)

# --- Step 6: Define Training Arguments ---
training_args = TrainingArguments(
    output_dir="./ner_model_results",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    logging_steps=10,
)

# --- Step 7: Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# --- Step 8: Train ---
print("--- Starting Training ---")
trainer.train()
print("✅ Training complete.")

# --- Step 9: Save Final Model ---
print(f"Saving model to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- Model saved successfully. ---")