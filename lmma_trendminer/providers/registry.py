# lmma_trendminer/providers/registry.py
from functools import lru_cache
from .custom_intent import TrainedIntentProvider
from .custom_ner import TrainedTokenNERProvider

_INTENT_PATH = "./models/intent-classifier"
_NER_PATH    = "./models/ner-extractor"

def configure_models(*, intent_path: str | None = None, ner_path: str | None = None) -> None:
    """Call this once at startup if your model paths differ from defaults."""
    global _INTENT_PATH, _NER_PATH
    if intent_path: _INTENT_PATH = intent_path
    if ner_path:    _NER_PATH    = ner_path
    # bust caches if already created
    get_intent.cache_clear()
    get_ner.cache_clear()

@lru_cache(maxsize=1)
def get_intent() -> TrainedIntentProvider:
    return TrainedIntentProvider(model_path=_INTENT_PATH)

@lru_cache(maxsize=1)
def get_ner() -> TrainedTokenNERProvider:
    return TrainedTokenNERProvider(model_path=_NER_PATH)
