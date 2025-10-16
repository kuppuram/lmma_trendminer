from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Dict, Any
from transformers import pipeline

class IntentProvider(ABC):
    @abstractmethod
    def predict(self, text: str, labels: List[str]) -> Dict[str, Any]: ...

@lru_cache(maxsize=1)
def _get_zero_shot_pipe(model_id: str = "facebook/bart-large-mnli"):
    # device=-1 = CPU; change to 0 for GPU
    return pipeline("zero-shot-classification", model=model_id, device=-1)

class ZeroShotMNLIIntent(IntentProvider):
    def __init__(self, model_id: str = "facebook/bart-large-mnli"):
        self.model_id = model_id
        # don’t create a new pipeline each time; reuse cached one
        self._pipe = _get_zero_shot_pipe(self.model_id)

    def predict(self, text: str, labels: List[str]) -> Dict[str, Any]:
        out = self._pipe(text, candidate_labels=labels, multi_label=False)
        return {"intent": out["labels"][0], "scores": dict(zip(out["labels"], out["scores"]))}
