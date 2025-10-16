# trendminer/pipeline/orchestrator.py
from typing import Dict, Any, List
from ..providers.intent import IntentProvider
from ..providers.ner import NERProvider
from ..providers.vector_store import VectorStore
from .retrieve import vector_only_retrieve
from .reduce import umap_reduce
from .cluster import hdbscan_cluster
from .label import select_exemplars, tfidf_labels
from lmma_trendminer.pipeline.metadata import build_metadata_from_query
from lmma_trendminer.pipeline.metadata_from_entities import build_metadata_from_entities


def map_entities_to_filters(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    for e in entities:
        if e.get("entity_group") in ("ORG","PRODUCT","MISC"):
            f.setdefault("product_keywords", []).append(e["word"].lower())
    return f

def run_trend_pipeline(
    query: str,
    intent_labels: List[str],
    intent: IntentProvider,
    ner: NERProvider,
    store: VectorStore,
    *,
    size: int = 2000,
    text_field: str = "text",
    umap_cfg: Dict[str, Any] = None,
    hdbscan_cfg: Dict[str, Any] = None
) -> Dict[str, Any]:

    # 1) parse
    # intent_res = intent.predict(query, intent_labels)
    # ents = ner.extract(query)
    # filters = map_entities_to_filters(ents)
    ents = ner.extract(query)
    meta_from_ents = build_metadata_from_entities(query, ents)
    print(meta_from_ents)

    filters = map_entities_to_filters(meta_from_ents["entities"])
    # print(filters)
    range_filters = {}
    if meta_from_ents.get("start_date") and meta_from_ents.get("end_date"):
        range_filters["timestamp"] = {
            "gte": meta_from_ents["start_date"],
            "lte": meta_from_ents["end_date"],
        }
    if meta_from_ents.get("score") is not None:
        range_filters["rating"] = {"gte": meta_from_ents["score"], "lte": meta_from_ents["score"]}
        
    # meta = build_metadata_from_query(query, intent, ner)
    # print(meta)
    # filters = map_entities_to_filters(meta["entities"])

    # # optional date → range filter (if your OpenSearch adapter supports it)
    # range_filters = None
    # if meta["start_date"] and meta["end_date"]:
    #     range_filters = {"timestamp": {"gte": meta["start_date"], "lte": meta["end_date"]}}
        

    # 2) vectors only

    # ids, X = vector_only_retrieve(store, size=2000, filters=filters)
    # ids, X = vector_only_retrieve(
    #     store,
    #     size=2000,
    #     filters=filters,
    #     range_filters=range_filters,
    # )
    print(filters, range_filters)
    ids, X = vector_only_retrieve(store, size=size, filters=filters, range_filters=range_filters)
  
    if len(ids) == 0:
        return {"clusters": [], "note": "no candidates"}

    if len(ids) < 10:  # threshold you like
        return {
            "intent": meta_from_ents["intent"],
            "intent_scores": meta_from_ents["intent_scores"],
            "entities": meta_from_ents["entities"],
            "total_vectors": len(ids),
            "noise_points": 0,
            "clusters": [],
            "note": f"Not enough candidates for clustering (got {len(ids)}). Try expanding date range or filters.",
        }

    # 3) reduce + cluster
    Z = umap_reduce(X, **(umap_cfg or {}))
    labels, probs, _ = hdbscan_cluster(Z, **(hdbscan_cfg or {}))

    # 4) exemplars + minimal mget
    idx_by_cluster = select_exemplars(Z, labels, k=5)
    cluster_docs: Dict[int, List[str]] = {}
    for c, idxs in idx_by_cluster.items():
        id_subset = [ids[i] for i in idxs]
        docs = store.mget(id_subset, fields=[text_field])
        cluster_docs[c] = [docs[i].get(text_field, "") for i in id_subset]

    # 5) labels
    label_map = tfidf_labels(cluster_docs, top_terms=5)

    # assemble
    out = []
    for c, idxs in idx_by_cluster.items():
        out.append({
            "cluster_id": c,
            "size": int((labels == c).sum()),
            "label": label_map.get(c, ""),
            "exemplar_ids": [ids[i] for i in idxs]
        })
    out.sort(key=lambda x: -x["size"])
    return {
        "intent": meta_from_ents["intent"],
        "intent_scores": meta_from_ents["intent_scores"],
        "entities": meta_from_ents["entities"],
        "total_vectors": len(ids),
        "noise_points": int((labels == -1).sum()),
        "clusters": out
    }
