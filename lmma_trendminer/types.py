from typing import TypedDict, List, Dict, Any

class IntentResult(TypedDict):
    intent: str
    scores: Dict[str, float]

class EntitySpan(TypedDict):
    entity_group: str
    word: str
    score: float
    start: int
    end: int

class VectorItem(TypedDict):
    id: str
    vector: List[float]
