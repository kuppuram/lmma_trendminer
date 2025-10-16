# Filename: train_ner_cli.py
import os
import typer
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification, # We need this for evaluation
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score # For computing metrics

app = typer.Typer()

@app.command()
def main(
    train_file: str = typer.Argument(None, help="Path to the training JSONL file."),
    eval_file: str = typer.Option(None, "--eval-file", "-ev", help="Optional path to the evaluation JSONL file."), # <-- ADDED
    output_dir: str = typer.Option("./models/ner-extractor", "--output-dir", "-o", help="Directory to save the trained model."),
    base_model: str = typer.Option("distilbert-base-uncased", "--base-model", "-m", help="Base Hugging Face model to fine-tune."),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs."), # Reduced default for faster runs with eval
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Training and evaluation batch size."),
):
    """
    Trains a NER model from a JSONL file, with an optional evaluation step.
    """
    typer.echo("--- Starting NER Model Training ---")

    # --- Step 1: Conditional Data Loading ---
    typer.echo(f"Loading dataset from {train_file}...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    
    ds = DatasetDict({"train": train_dataset})

    if eval_file:
        typer.echo(f"Loading evaluation dataset from {eval_file}...")
        eval_dataset = load_dataset("json", data_files=eval_file, split="train")
        ds["validation"] = eval_dataset
    
    typer.echo(f"✅ Datasets loaded: {list(ds.keys())}")

    # --- Step 2: Infer Labels ---
    typer.echo("Inferring labels from the training dataset...")
    uniq = set()
    for row in ds["train"]:
        uniq.update(row['labels'])
    labels_list = sorted(list(uniq))
    id2label = {i: l for i, l in enumerate(labels_list)}
    label2id = {l: i for i, l in enumerate(labels_list)}
    typer.echo(f"✅ Labels inferred: {labels_list}")

    # --- Step 3: Initialize Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer) # <-- ADDED for padding batches correctly

    # --- Step 4: Tokenize and Align ---
    # The function itself is unchanged
    def tokenize_and_align_labels(example):
        tokenized_input = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        word_ids = tokenized_input.word_ids()
        word_label_ids = [label2id[l] for l in example["labels"]]
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None: aligned_labels.append(-100)
            elif word_idx != previous_word_idx: aligned_labels.append(word_label_ids[word_idx])
            else: aligned_labels.append(-100)
            previous_word_idx = word_idx
        tokenized_input["labels"] = aligned_labels
        return tokenized_input

    typer.echo("Tokenizing and aligning labels...")
    tokenized_ds = ds.map(tokenize_and_align_labels, remove_columns=ds["train"].column_names)
    typer.echo("✅ Tokenization complete.")
    
    # --- Step 5: Define Metrics Calculation ---
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        f1 = f1_score(true_labels, true_predictions)
        return {"f1": f1}

    # --- Step 6: Load Model ---
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(labels_list), id2label=id2label, label2id=label2id
    )

    # --- Step 7: Update Training Arguments ---
    # --- Current, modern code (transformers accelerate) ---

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, # <-- ADDED for eval
        logging_steps=10,
        # --- ADDED EVALUATION STRATEGY ---
        eval_strategy="epoch" if eval_file else "no",
        save_strategy="epoch" if eval_file else "steps",
        load_best_model_at_end=True if eval_file else False,
    )

    # --- Alternative, older code (transformers accelerate) ---
    # training_args = TrainingArguments(
    #     output_dir=f"{output_dir}/results",
    #     num_train_epochs=epochs,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     logging_steps=10,
    #     # The old way to enable evaluation
    #     evaluate_during_training=True if eval_file else False,
    #     # The old way to save at each evaluation
    #     save_steps=10, # Match logging_steps or another value
    #     load_best_model_at_end=True if eval_file else False,
    # )

    # --- Step 8: Update Trainer Initialization ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get("validation"), # <-- ADDED (use .get for safety)
        tokenizer=tokenizer, # For saving tokenizer with model
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_file else None, # <-- ADDED
    )

    # --- Step 9: Train and Save ---
    typer.echo("--- Starting Training ---")
    trainer.train()
    typer.echo("✅ Training complete.")
    typer.echo(f"Saving final model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    typer.echo("--- Model saved successfully. ---")

if __name__ == "__main__":
    app()