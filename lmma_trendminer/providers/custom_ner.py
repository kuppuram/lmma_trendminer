# lmma_trendminer/providers/custom_ner.py
from functools import lru_cache
from typing import List, Dict, Any
from transformers import pipeline, AutoConfig

# Simple heuristics to coerce unknown LABEL_* to our schema
def _guess_entity_group(word: str) -> str | None:
    w = word.lower().strip(".")
    months = {"jan","january","feb","february","mar","march","apr","april","may","jun","june",
              "jul","july","aug","august","sep","sept","september","oct","october",
              "nov","november","dec","december"}
    if w in months:
        return "MONTH"
    if w.isdigit() and len(w) == 4:
        return "YEAR"
    if w.isdigit() and len(w) == 1:
        return "SCORE"
    return None

def _normalize_group(g: str) -> str:
    # Strip BIO prefixes; map LABEL_* to empty, we’ll guess later
    if g.startswith("B-") or g.startswith("I-"):
        g = g.split("-", 1)[1]
    if g.startswith("LABEL_"):
        return ""  # unknown; will guess from token text
    return g

@lru_cache(maxsize=1)
def _ner_pipe(model_path: str):
    # Load both model & tokenizer from the same folder to keep label maps aligned
    pipe = pipeline(
        task="token-classification",
        model=model_path,
        tokenizer=model_path,
        # aggregation_strategy="simple",
        aggregation_strategy="none",  # <-- TEMPORARILY CHANGE THIS
        device=-1,  # CPU; set to 0 for CUDA
    )
    # Helpful one-time debug: what labels did we load?
    try:
        cfg = AutoConfig.from_pretrained(model_path)
        print("[NER] id2label:", getattr(cfg, "id2label", None))
        print("[NER] label2id:", getattr(cfg, "label2id", None))
    except Exception:
        pass
    return pipe

class TrainedTokenNERProvider:
    """
    Wraps your trained token classifier.
    Works even if model emits BIO groups or generic LABEL_*.
    """
    def __init__(self, model_path: str = "./models/ner-extractor"):
        self.model_path = model_path
        self.pipe = _ner_pipe(self.model_path)

    def extract(self, text: str) -> List[Dict[str, Any]]:
        raw = self.pipe(text)  # aggregated spans
        print(f"[RAW TOKEN PREDICTIONS]: {raw}") # <-- ADD THIS LINE
        out: List[Dict[str, Any]] = []
        for e in raw:
            g = _normalize_group(e.get("entity_group", ""))
            if not g:
                # Try to guess if model returned LABEL_* or unknown group
                guessed = _guess_entity_group(e.get("word", ""))
                if guessed:
                    g = guessed
            if g:
                e2 = dict(e)
                e2["entity_group"] = g  # normalized
                out.append(e2)
        return out
