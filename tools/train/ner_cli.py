# tools/train/ner_cli.py
import os
from typing import List, Optional

import typer
import yaml
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from seqeval.metrics import f1_score

# fallback to your local python that returns BIO samples
try:
    from ner_data import get_ner_data  # your function
except Exception:
    get_ner_data = None

app = typer.Typer(help="Train a token-classification (NER) model")

def make_training_args(output_dir, lr, batch_size, epochs, weight_decay, fp16, seed):
    # Minimal, version-agnostic TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        report_to=[],
        seed=seed,
        fp16=fp16,
    )

def _load_from_jsonl(train_file: str, eval_file: str) -> DatasetDict:
    # expects each JSON line to have: {"tokens":[...], "labels":[...]}
    ds_train = load_dataset("json", data_files=train_file, split="train")
    ds_eval = load_dataset("json", data_files=eval_file, split="train")
    return DatasetDict({"train": ds_train, "validation": ds_eval})

def _load_from_py() -> DatasetDict:
    if not get_ner_data:
        raise RuntimeError("ner_data.get_ner_data() not found. Provide --train_file/--eval_file JSONL instead.")
    data = get_ner_data()
    # build Dataset from your python objects
    train = Dataset.from_list([{"tokens": d["tokens"], "labels": d["labels"]} for d in data["train"]])
    val   = Dataset.from_list([{"tokens": d["tokens"], "labels": d["labels"]} for d in data["validation"]])
    return DatasetDict({"train": train, "validation": val})

# def _align_labels_with_tokens(word_labels: List[int], word_ids: List[Optional[int]]) -> List[int]:
#     aligned = []
#     prev = None
#     for wid in word_ids:
#         if wid is None:
#             aligned.append(-100)
#         elif wid != prev:
#             aligned.append(word_labels[wid])
#         else:
#             aligned.append(word_labels[wid])
#         prev = wid
#     return aligned

def _align_labels_with_tokens(word_labels: List[int], word_ids: List[Optional[int]]) -> List[int]:
    aligned = []
    prev = None
    for wid in word_ids:
        if wid is None:
            aligned.append(-100)
        elif wid != prev:
            aligned.append(word_labels[wid])
        else:
            # aligned.append(word_labels[wid])
            # CORRECT: Ignore subsequent sub-tokens of the same word
            aligned.append(-100)
        prev = wid
    return aligned

@app.command()
def run(
    # Config (YAML) is optional; you can also pass flags directly
    config: Optional[str] = typer.Option(None, help="YAML with model/output/labels, like tools/train/configs/ner-bert.yaml"),
    # Direct flags (override or used when no YAML)
    base_model: str = typer.Option("distilbert-base-uncased", help="HF model name or local path"),
    output_dir: str = typer.Option("./models/ner-extractor", help="Where to save the trained model"),
    labels_csv: Optional[str] = typer.Option(None, help="Comma-separated labels if not using YAML"),
    train_file: Optional[str] = typer.Option(None, help="JSONL file with tokens+labels"),
    eval_file: Optional[str] = typer.Option(None, help="JSONL file with tokens+labels"),
    epochs: int = typer.Option(5, help="Training epochs"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    batch_size: int = typer.Option(16, help="Per-device batch size"),
    warmup_ratio: float = typer.Option(0.1, help="Warmup ratio"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    seed: int = typer.Option(42, help="Random seed"),
    fp16: bool = typer.Option(False, help="Use fp16 (requires GPU/AMP)"),
):
    # Load YAML if provided
    if config:
        with open(config, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        base_model = y.get("model_name", base_model)
        output_dir = y.get("output_dir", output_dir)
        labels = y.get("labels")
        epochs = y.get("hyper", {}).get("epochs", epochs)
        lr = y.get("hyper", {}).get("lr", lr)
        batch_size = y.get("hyper", {}).get("batch_size", batch_size)
        warmup_ratio = y.get("hyper", {}).get("warmup_ratio", warmup_ratio)
        weight_decay = y.get("hyper", {}).get("weight_decay", weight_decay)
        seed = y.get("misc", {}).get("seed", seed)
        fp16 = y.get("misc", {}).get("fp16", fp16)
        train_file = y.get("train_file", train_file)
        eval_file = y.get("eval_file", eval_file)
    else:
        # labels = [s.strip() for s in (labels_csv or "O,B-MONTH,I-MONTH,YEAR,DATE,SCORE,TOPK").split(",")]
         # This string MUST contain every possible label from your ner_data.py
        labels = [s.strip() for s in (labels_csv or "O,B-DATE,B-MONTH,B-SCORE,B-YEAR,I-DATE,I-MONTH").split(",")]

    # set_seed(seed)
    # --- START OF REFACTORED LOGIC ---
    set_seed(seed)

    # Load dataset
    # if train_file and eval_file:
    #     ds = _load_from_jsonl(train_file, eval_file)
    # else:
    #     ds = _load_from_py()

    if train_file:
        ds = DatasetDict({
            "train": load_dataset("json", data_files=train_file, split="train")
        })
    else:
        # Fallback for ner_data.py is more complex, focus on jsonl path
        raise ValueError("Please provide a --train-file.")

    # --- REFACTORED LABEL LOGIC ---
    if labels_csv:
        # 1. Highest priority: Use the user-provided CSV string
        labels = [s.strip() for s in labels_csv.split(",")]
    elif not config:
        # 2. If no config or CSV, infer labels directly from the loaded dataset
        print("Inferring labels from the dataset...")
        uniq = set()
        # for split in ["train", "validation"]:
        #     for row in ds[split]:
        #         uniq.update(row["labels"])
        for row in ds["train"]:
            uniq.update(row["labels"])
        labels = sorted(list(uniq))
    # 3. Lowest priority: 'labels' from a YAML config would already be set

    print(f"Using labels: {labels}")
    # --- END OF REFACTORED LOGIC ---

    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    # Tokenizer & tokenize+align
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # ADD THIS NEW, CORRECT FUNCTION in its place
    def tokenize_and_align_single(example):
        # Tokenize the pre-split words
        tokenized_input = tok(example["tokens"], is_split_into_words=True, truncation=True)
        
        # Get the integer IDs for the string labels
        word_label_ids = [label2id[l] for l in example["labels"]]
        
        # Align the labels with the new sub-word tokens
        aligned_labels = _align_labels_with_tokens(word_label_ids, tokenized_input.word_ids())
        
        tokenized_input["labels"] = aligned_labels
        return tokenized_input

    # New, correct .map() call
    ds_tok = ds.map(tokenize_and_align_single, remove_columns=ds["train"].column_names)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tok)

    # Training
    # args = TrainingArguments(
    #     output_dir=output_dir,
    #     learning_rate=lr,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=epochs,
    #     weight_decay=weight_decay,
    #     warmup_ratio=warmup_ratio,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     logging_steps=50,
    #     fp16=fp16,
    #     report_to=[],
    #     seed=seed,
    # )

    args = make_training_args(
        output_dir=output_dir,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        fp16=fp16,
        seed=seed,
    )

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        out_label_ids = p.label_ids
        true_preds, true_labels = [], []
        for pred, lab in zip(preds, out_label_ids):
            true_pred = [labels[p] for (p, l) in zip(pred, lab) if l != -100]
            true_lab  = [labels[l] for (p, l) in zip(pred, lab) if l != -100]
            true_preds.append(true_pred)
            true_labels.append(true_lab)
        return {"f1": f1_score(true_labels, true_preds)}

    # # trainer = Trainer(
    # #     model=model,
    # #     args=args,
    # #     train_dataset=ds_tok["train"],
    # #     eval_dataset=ds_tok["validation"],
    # #     tokenizer=tok,
    # #     data_collator=data_collator,
    # #     compute_metrics=compute_metrics,
    # # )
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=ds_tok["train"],
    #     eval_dataset=ds_tok["validation"],
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        # No eval_dataset or compute_metrics
        data_collator=data_collator,
    )

    trainer.train()
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)

    typer.echo(f"Saved NER model to {output_dir}\nLabels: {labels}")

if __name__ == "__main__":
    app()
