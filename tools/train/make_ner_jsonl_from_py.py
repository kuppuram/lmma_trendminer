import json, random
from typing import List, Dict
import typer

app = typer.Typer(help="Convert get_ner_data() (text + space-separated labels) to JSONL with tokens/labels arrays")

def _load_py(module_name: str) -> List[Dict]:
    mod = __import__(module_name, fromlist=["get_ner_data"])
    data = mod.get_ner_data()
    if not isinstance(data, list):
        raise ValueError("Expected get_ner_data() -> List[dict] with keys 'text' and 'labels'")
    return data

@app.command()
def run(
    module: str = typer.Argument("ner_data", help="Python module that defines get_ner_data()"),
    train_out: str = typer.Option("./data/ner_train.jsonl"),
    eval_out: str  = typer.Option("./data/ner_eval.jsonl"),
    split_ratio: float = typer.Option(0.8, help="Train split ratio"),
    seed: int = typer.Option(42),
):
    random.seed(seed)
    rows = _load_py(module)

    # Convert to {tokens: [...], labels: [...]} per row
    examples = []
    for i, row in enumerate(rows):
        text = row["text"].strip()
        label_str = row["labels"].strip()
        tokens = text.split()                 # simple whitespace tokenization (matches your labeling)
        labels = label_str.split()
        if len(tokens) != len(labels):
            raise ValueError(f"tokens/labels length mismatch at row {i}: "
                             f"{len(tokens)} tokens vs {len(labels)} labels.\ntext={text}\nlabels={label_str}")
        examples.append({"tokens": tokens, "labels": labels})

    random.shuffle(examples)
    n_train = int(len(examples) * split_ratio)
    train, val = examples[:n_train], examples[n_train:]

    def dump(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex) + "\n")

    dump(train_out, train)
    dump(eval_out, val)
    typer.echo(f"Wrote {train_out} (n={len(train)}) and {eval_out} (n={len(val)})")

if __name__ == "__main__":
    app()
