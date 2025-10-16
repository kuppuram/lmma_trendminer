from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Dict, Any
from transformers import pipeline

class NERProvider(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[Dict[str, Any]]: ...

@lru_cache(maxsize=1)
def _get_ner_pipe(model_id: str = "dslim/bert-base-NER"):
    return pipeline("token-classification", model=model_id, aggregation_strategy="simple", device=-1)

class BertBaseNER(NERProvider):
    def __init__(self, model_id: str = "dslim/bert-base-NER"):
        self.model_id = model_id
        self._pipe = _get_ner_pipe(self.model_id)

    def extract(self, text: str) -> List[Dict[str, Any]]:
        return self._pipe(text)
