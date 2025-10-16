# trendminer/pipeline/retrieve.py
# from typing import Tuple, List, Dict, Any
# import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
# from typing import Tuple, Dict, Any, Optional
from ..providers.vector_store import VectorStore

# def vector_only_retrieve(store: VectorStore, size: int, filters: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
#     items = store.get_vectors(size=size, filters=filters)
# def vector_only_retrieve(store: VectorStore, size: int, filters: Dict[str, Any], range_filters: Dict[str, Dict[str, Any]] | None = None):
#     items = store.get_vectors(size=size, filters=filters, range_filters=range_filters)
#     ids = [x["id"] for x in items]
#     X = np.array([x["vector"] for x in items], dtype="float32")
#     return ids, X

# def vector_only_retrieve(
#     store,
#     size: int,
#     filters: Optional[Dict[str, Any]] = None,
#     range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
# ) -> Tuple[list[str], np.ndarray]:
#     items = store.get_vectors(size=size, filters=filters, range_filters=range_filters)
#     ids = [it["id"] for it in items]
#     X = np.array([it["vector"] for it in items], dtype=np.float32)
#     return ids, X

# def vector_only_retrieve(
#     store,
#     size: int,
#     filters: Optional[Dict[str, Any]] = None,
#     range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
#     extra_filter_clauses: Optional[List[Dict[str, Any]]] = None,  # NEW
#     must_clauses: Optional[List[Dict[str, Any]]] = None,          # NEW
# ) -> Tuple[List[str], np.ndarray]:
#     items = store.get_vectors(
#         size=size,
#         filters=filters,
#         range_filters=range_filters,
#         extra_filter_clauses=extra_filter_clauses,  # NEW
#         must_clauses=must_clauses,                  # NEW
#     )
#     ids = [it["id"] for it in items]
#     X = np.array([it["vector"] for it in items], dtype=np.float32)
#     return ids, X

def vector_only_retrieve(
    store,
    size: int,
    filters: Optional[Dict[str, Any]] = None,
    range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    extra_filter_clauses: Optional[List[Dict[str, Any]]] = None,  # NEW
    must_clauses: Optional[List[Dict[str, Any]]] = None,          # NEW
) -> Tuple[List[str], np.ndarray]:
    items = store.get_vectors(
        size=size,
        filters=filters,
        range_filters=range_filters,
        extra_filter_clauses=extra_filter_clauses,
        must_clauses=must_clauses,
    )
    ids = [it["id"] for it in items]
    X = np.array([it["vector"] for it in items], dtype=np.float32)
    return ids, X