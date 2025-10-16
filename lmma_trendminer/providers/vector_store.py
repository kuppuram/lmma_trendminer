# trendminer/providers/vector_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..types import VectorItem
from typing import List, Dict, Any, Optional
from typing import Optional, Dict, Any, List

VectorItem = Dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def get_vectors(self, size: int, filters: Dict[str, Any]) -> List[VectorItem]: ...
    @abstractmethod
    def mget(self, ids: List[str], fields: List[str] | None = None) -> Dict[str, Dict[str, Any]]: ...

# class OpenSearchStore(VectorStore):
#     # def __init__(self, client, index: str, vector_field: str = "vector_embedding"):
#     def __init__(self, client, index: str, vector_field: str = "review_embedding"):
#         self.client = client
#         self.index = index
#         self.vector_field = vector_field

#     # def get_vectors(self, size: int, filters: Dict[str, Any]) -> List[VectorItem]:
#     #     body = {
#     #         "size": size,
#     #         "_source": [self.vector_field],
#     #         "query": {"bool": {"filter": [{"term": {k: v}} for k, v in (filters or {}).items()]}} if filters else {"match_all": {}}
#     #     }
#     #     resp = self.client.search(index=self.index, body=body)
#     #     out = []
#     #     for h in resp.get("hits", {}).get("hits", []):
#     #         vec = h.get("_source", {}).get(self.vector_field)
#     #         if vec is not None:
#     #             out.append({"id": h["_id"], "vector": vec})
#     #     return out

#     def get_vectors(
#         self,
#         size: int,
#         filters: Optional[Dict[str, Any]] = None,
#         range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
#     ) -> List[VectorItem]:
#         clauses: List[Dict[str, Any]] = []
#         if filters:
#             clauses += [{"term": {k: v}} for k, v in filters.items()]
#         if range_filters:
#             for field, bounds in range_filters.items():
#                 clauses.append({"range": {field: bounds}})

#         body = {
#             "size": size,
#             "_source": [self.vector_field],
#             "query": {"bool": {"filter": clauses}} if clauses else {"match_all": {}},
#         }
#         resp = self.client.search(index=self.index, body=body)
#         hits = resp.get("hits", {}).get("hits", [])
#         return [
#             {"id": h["_id"], "vector": h["_source"][self.vector_field]}
#             for h in hits
#             if h.get("_source", {}).get(self.vector_field) is not None
#         ]
    
#     def mget(self, ids: List[str], fields: List[str] | None = None) -> Dict[str, Dict[str, Any]]:
#         """
#         Works across OpenSearch versions:
#         - Use 'docs' form + _source_includes query param (most compatible).
#         - Falls back to 'ids' form if needed.
#         """
#         params = None
#         if fields:
#             # Comma-separated list for _source filtering
#             params = {"_source_includes": ",".join(fields)}

#         # Preferred: 'docs' shape (explicit index per doc)
#         body = {"docs": [{"_index": self.index, "_id": _id} for _id in ids]}
#         try:
#             resp = self.client.mget(body=body, params=params)
#         except Exception:
#             # Fallback: 'ids' shape (when index is provided in the path)
#             body = {"ids": ids}
#             resp = self.client.mget(index=self.index, body=body, params=params)

#         out: Dict[str, Dict[str, Any]] = {}
#         for d in resp.get("docs", []):
#             if d.get("found"):
#                 out[d["_id"]] = d.get("_source", {}) or {}
#         return out

# class OpenSearchStore:
#     def __init__(self, client, index: str, vector_field: str):
#         self.client = client
#         self.index = index
#         self.vector_field = vector_field

#     def get_vectors(
#         self,
#         size: int,
#         filters: Optional[Dict[str, Any]] = None,
#         range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
#         extra_filter_clauses: Optional[List[Dict[str, Any]]] = None,  # NEW
#         must_clauses: Optional[List[Dict[str, Any]]] = None,          # NEW
#     ) -> List[VectorItem]:
#         # Build bool clauses
#         filter_clauses: List[Dict[str, Any]] = []
#         if filters:
#             filter_clauses += [{"term": {k: v}} for k, v in filters.items()]
#         if range_filters:
#             for field, bounds in range_filters.items():
#                 filter_clauses.append({"range": {field: bounds}})
#         if extra_filter_clauses:
#             filter_clauses += list(extra_filter_clauses)

#         bool_q: Dict[str, Any] = {}
#         if filter_clauses:
#             bool_q["filter"] = filter_clauses
#         if must_clauses:
#             bool_q["must"] = list(must_clauses)

#         body = {
#             "size": size,
#             "_source": [self.vector_field],  # only fetch the vector
#             "query": {"bool": bool_q} if bool_q else {"match_all": {}},
#         }
#         resp = self.client.search(index=self.index, body=body)
#         hits = resp.get("hits", {}).get("hits", [])
#         out: List[VectorItem] = []
#         for h in hits:
#             src = h.get("_source", {}) or {}
#             vec = src.get(self.vector_field)
#             if vec is not None:
#                 out.append({"id": h["_id"], "vector": vec})
#         return out

#     def mget(self, ids: List[str], fields: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
#         """
#         Fetch selected fields for the given IDs. Uses _source_includes for compatibility.
#         Returns a dict: {id: {_source...}, ...}
#         """
#         params = None
#         if fields:
#             params = {"_source_includes": ",".join(fields)}
#         # docs-shape is the most compatible across OS versions
#         body = {"docs": [{"_index": self.index, "_id": _id} for _id in ids]}
#         resp = self.client.mget(body=body, params=params)
#         out: Dict[str, Dict[str, Any]] = {}
#         for d in resp.get("docs", []):
#             if d.get("found"):
#                 out[d["_id"]] = d.get("_source", {}) or {}
#         return out

class OpenSearchStore:
    def __init__(self, client, index: str, vector_field: str):
        self.client = client
        self.index = index
        self.vector_field = vector_field

    def get_vectors(
        self,
        size: int,
        filters: Optional[Dict[str, Any]] = None,
        range_filters: Optional[Dict[str, Dict[str, Any]]] = None,
        extra_filter_clauses: Optional[List[Dict[str, Any]]] = None,  # NEW
        must_clauses: Optional[List[Dict[str, Any]]] = None,          # NEW
    ) -> List[VectorItem]:
        """
        Build a bool query:
          - 'filter' holds term/range/extra prebuilt clauses (no scoring)
          - 'must'   holds full-text clauses (e.g., multi_match), optional
        Returns a list of {"id": <doc_id>, "vector": <embedding list>}
        """
        # 1) collect filter clauses
        filter_clauses: List[Dict[str, Any]] = []
        if filters:
            # simple equality filters -> 'term'
            filter_clauses += [{"term": {k: v}} for k, v in filters.items()]
        if range_filters:
            # date/score ranges -> 'range'
            for field, bounds in range_filters.items():
                filter_clauses.append({"range": {field: bounds}})
        if extra_filter_clauses:
            # already-formed OS clauses (e.g., 'terms' for discrete score expansion)
            filter_clauses += list(extra_filter_clauses)

        # 2) assemble bool query
        bool_q: Dict[str, Any] = {}
        if filter_clauses:
            bool_q["filter"] = filter_clauses
        if must_clauses:
            # text queries (e.g., multi_match on ["text","title"])
            bool_q["must"] = list(must_clauses)

        # 3) final request body: only fetch the vector field to keep it fast
        body = {
            "size": size,
            "_source": [self.vector_field],  # CRITICAL: Only return the vector
            "query": {"bool": bool_q} if bool_q else {"match_all": {}},
        }

        # 4) call OpenSearch
        print("[DEBUG OS QUERY]", body)
        resp = self.client.search(index=self.index, body=body)
        hits = resp.get("hits", {}).get("hits", [])

        # 5) convert to [{"id":..., "vector":[...]}]
        out: List[VectorItem] = []
        for h in hits:
            src = h.get("_source") or {}
            vec = src.get(self.vector_field)
            if vec is not None:
                out.append({"id": h["_id"], "vector": vec})
        return out

    def mget(self, ids: List[str], fields: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fetch selected fields for the given IDs. Uses _source_includes for compatibility.
        Returns {id: {_source...}, ...}
        """
        params = None
        if fields:
            params = {"_source_includes": ",".join(fields)}
        body = {"docs": [{"_index": self.index, "_id": _id} for _id in ids]}
        resp = self.client.mget(body=body, params=params)
        out: Dict[str, Dict[str, Any]] = {}
        for d in resp.get("docs", []):
            if d.get("found"):
                out[d["_id"]] = d.get("_source", {}) or {}
        return out

