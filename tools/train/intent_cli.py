# tools/train/intent_cli.py
import os
from dataclasses import dataclass
from typing import Optional, List

import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np

app = typer.Typer(help="Train an intent classifier (text → intent label)")

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

@dataclass
class TrainCfg:
    base_model: str = "distilbert-base-uncased"
    output_dir: str = "./models/intent-classifier"
    epochs: int = 5
    lr: float = 5e-5
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    
def _build_dataset(csv_path: str) -> tuple[DatasetDict, List[str]]:
    # expects columns: text,label
    df = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text,label")

    # label encode
    labels = sorted(df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    df["label_id"] = df["label"].map(label2id)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "label_id"]], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[["text", "label_id"]], preserve_index=False),
    })
    return ds, labels

@app.command()
def run(
    csv_path: str = typer.Argument(..., help="Path to intent CSV with columns: text,label"),
    base_model: str = typer.Option("distilbert-base-uncased", help="HF model name or local path"),
    output_dir: str = typer.Option("./models/intent-classifier", help="Where to save the trained model"),
    epochs: int = typer.Option(5, help="Training epochs"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    batch_size: int = typer.Option(16, help="Per-device batch size"),
    warmup_ratio: float = typer.Option(0.1, help="Warmup ratio"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    seed: int = typer.Option(42, help="Random seed"),
    fp16: bool = typer.Option(False, help="Use mixed precision if supported"),
):
    cfg = TrainCfg(
        base_model=base_model, output_dir=output_dir, epochs=epochs, lr=lr,
        batch_size=batch_size, warmup_ratio=warmup_ratio, weight_decay=weight_decay, seed=seed
    )

    ds, labels = _build_dataset(csv_path)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, padding=False)
        out["labels"] = batch["label_id"]
        return out

    ds_tok = ds.map(tok, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    # args = TrainingArguments(
    #     output_dir=cfg.output_dir,
    #     learning_rate=cfg.lr,
    #     per_device_train_batch_size=cfg.batch_size,
    #     per_device_eval_batch_size=cfg.batch_size,
    #     num_train_epochs=cfg.epochs,
    #     weight_decay=cfg.weight_decay,
    #     warmup_ratio=cfg.warmup_ratio,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     logging_steps=50,
    #     report_to=[],
    #     seed=cfg.seed,
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
        labels_np = p.label_ids
        acc = (preds == labels_np).mean().item()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    typer.echo(f"Saved intent model to {cfg.output_dir}\nLabels: {labels}")

if __name__ == "__main__":
    app()
