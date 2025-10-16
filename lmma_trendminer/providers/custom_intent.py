# lmma_trendminer/providers/custom_intent.py
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@lru_cache(maxsize=1)
def _intent_pipe(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
    # device=-1 for CPU; set 0 for GPU
    return pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True, device=-1)

class TrainedIntentProvider:
    """
    Wraps your trained intent classifier.
    Expects labels like: trend_analysis, simple_search, greeting.
    """
    def __init__(self, model_path: str = "../models/intent-classifier"):
        self.model_path = model_path
        self.pipe = _intent_pipe(self.model_path)

    def predict(self, text: str) -> dict:
        rows = self.pipe(text)[0]  # list of {'label': 'trend_analysis', 'score': 0.9}, ...
        scores = {r["label"]: float(r["score"]) for r in rows}
        intent = max(scores, key=scores.get) if scores else None
        return {"intent": intent, "scores": scores}

# lmma_trendminer/providers/custom_intent.py
# from transformers import pipeline
# from typing import Dict, Any, List

# class TrainedIntentProvider:
#     def __init__(self, model_path: str):
#         self.pipe = pipeline(
#             task="text-classification",
#             model=model_path,
#             tokenizer=model_path,
#             top_k=None,     # equivalent to old return_all_scores=True
#             device=-1
#         )

#     def predict(self, text: str, labels: List[str]) -> Dict[str, Any]:
#         res = self.pipe(text)[0]  # list of dicts [{"label": "...", "score": ...}, ...]
#         scores = {r["label"]: float(r["score"]) for r in res if r["label"] in labels}
#         # pick max over the provided label set (fallback if missing)
#         intent = max(scores, key=scores.get) if scores else (res[0]["label"] if res else "trend_analysis")
#         return {"intent": intent, "scores": scores}
