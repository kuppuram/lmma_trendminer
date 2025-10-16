# lmma_trendminer/runner.py
from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np

from .providers.registry import get_intent, get_ner
from .pipeline.metadata_from_entities import build_metadata_from_entities
from .providers.vector_store import OpenSearchStore  # or your interface
from .pipeline.retrieve import vector_only_retrieve
from .pipeline.reduce import umap_reduce
from .pipeline.cluster import hdbscan_cluster
from .pipeline.label import select_exemplars, tfidf_labels
from typing import Any, Callable, Dict, List, Tuple, Optional
from .config import load_config, ProfileCfg

# def _score_filter_clause(bounds: dict, score_cfg) -> dict:
#     lo = bounds.get("gte", score_cfg.min)
#     hi = bounds.get("lte", score_cfg.max)
#     if score_cfg.type == "numeric":
#         return {"range": {score_cfg.name: {k: v for k, v in bounds.items() if k in ("gte", "lte")}}}
#     else:
#         # keyword: expand to discrete terms
#         vals = list(range(lo, hi + 1))
#         return {"terms": {score_cfg.name: vals}}

# def _score_filter_clause(bounds: dict, score_cfg) -> dict:
#     lo = bounds.get("gte", score_cfg.min)
#     hi = bounds.get("lte", score_cfg.max)
#     if score_cfg.type == "numeric":
#         return {"range": {score_cfg.name: {k: v for k, v in bounds.items() if k in ("gte","lte")}}}
#     else:
#         # keyword field: expand to discrete values
#         return {"terms": {score_cfg.name: list(range(lo, hi+1))}}

# def _score_filter_clause(bounds: dict, score_cfg) -> dict:
#     """
#     bounds: {"gte": N} or {"lte": N} or {"gte": N, "lte": M}
#     score_cfg: from config (fields.score) with .name, .type ("numeric"|"keyword"),
#                and optionally .min, .max, .cast ("int"|"string")
#     """
#     # 1) normalize bounds
#     lo = int(bounds.get("gte", getattr(score_cfg, "min", 0)))
#     hi = int(bounds.get("lte", getattr(score_cfg, "max", 5)))
#     if lo > hi:
#         lo, hi = hi, lo  # swap if accidentally inverted

#     # 2) numeric field → use range
#     if getattr(score_cfg, "type", "numeric") == "numeric":
#         q = {}
#         if "gte" in bounds: q["gte"] = lo
#         if "lte" in bounds: q["lte"] = hi
#         return {"range": {score_cfg.name: q}}

#     # 3) keyword field → expand to discrete terms
#     vals = list(range(lo, hi + 1))
#     # Optional casting: many CSV ingests store scores as strings in keyword fields
#     cast = getattr(score_cfg, "cast", "int")  # you can add `cast: "string"` in YAML if needed
#     if cast == "string":
#         vals = [str(v) for v in vals]

#     return {"terms": {score_cfg.name: vals}}

def _score_filter_clause(bounds: dict, score_cfg) -> dict:
    lo = int(bounds.get("gte", getattr(score_cfg, "min", 0)))
    hi = int(bounds.get("lte", getattr(score_cfg, "max", 5)))
    if lo > hi:
        lo, hi = hi, lo

    if getattr(score_cfg, "type", "numeric") == "numeric":
        q = {}
        if "gte" in bounds: q["gte"] = lo
        if "lte" in bounds: q["lte"] = hi
        return {"range": {score_cfg.name: q}}

    vals = list(range(lo, hi + 1))
    if getattr(score_cfg, "cast", "int") == "string":
        vals = [str(v) for v in vals]
    return {"terms": {score_cfg.name: vals}}


def _must_clauses_from_query(query: str, text_fields: list[str], term_map: dict, synonyms: dict) -> list[dict]:
    q = query.lower()
    terms = []
    for key, base_terms in term_map.items():
        if key in q:
            bucket = set(base_terms)
            bucket.update(synonyms.get(key, []))
            terms.extend(list(bucket))
    if not terms:
        return []
    return [{
        "multi_match": {
            "query": " ".join(terms),
            "fields": text_fields,
            "type": "best_fields"
        }
    }]

def _term_filters_from_query(query: str) -> dict:
    q = query.lower()
    # Example: map a few coarse categories; extend as needed
    if "electronics" in q.lower():
        return {"category": "electronics"}
    if "food" in q.lower() or "foods" in q.lower() or "grocery" in q.lower():
        return {"category": "foods"}  # or "food"
    return {}

def analyze_query_and_summarize(
    query: str,
    store,
    *,
    profile: str = "amazon_food_reviews",   # <-- choose which profile to use
    min_reviews: int = 2000,
    text_field: str | None = None,          # deprecated, prefer profile.fields.text
    umap_cfg: Dict[str, Any] | None = None,
    hdbscan_cfg: Dict[str, Any] | None = None,
    summarizer: Optional[Callable[[List[str]], str]] = None,
) -> Tuple[str, Dict[str, Any]]:

    cfg = load_config()
    prof: ProfileCfg = cfg.profiles[profile]
    f = prof.fields

    # 1) intent + entities (same as before) ...
    from .providers.registry import get_intent, get_ner
    from .pipeline.metadata_from_entities import build_metadata_from_entities

    intent_res = get_intent().predict(query)
    if intent_res["intent"] != "trend_analysis":
        return ("I can only analyze trends right now. Please ask about 'top trends'.",
                {"intent": intent_res["intent"], "clusters": []})

    ents = get_ner().extract(query)
    print("[DEBUG NER OUTPUT]", ents)
    meta = build_metadata_from_entities(query, ents)


    # --- NORMALIZE META KEYS (drop this block in runner.py) ---
    rating_bounds = meta.get("rating")
    if rating_bounds is None:
        if isinstance(meta.get("score_range"), dict):
            rating_bounds = meta["score_range"]
        elif meta.get("score") is not None:
            s = int(meta["score"])
            rating_bounds = {"gte": s, "lte": s}

    if rating_bounds is not None:
        meta["rating"] = rating_bounds
        # optional cleanup so 'details.meta' shows a single source of truth:
        meta.pop("score", None)
        meta.pop("score_range", None)
    # --- end normalize block ---
    print("[DEBUG META]", meta)
    if "start_date" not in meta or "end_date" not in meta:
        return ("Could not determine a date range from your query.",
                {"intent": intent_res["intent"], "entities": ents, "meta": meta, "clusters": []})

    # 2) Build OS query pieces from config
    range_filters = { f.timestamp: {"gte": meta["start_date"], "lte": meta["end_date"]} }
    term_filters  = {}  # if you map entities→terms later

    extra_filter_clauses = []
    if meta.get("rating"):                             # bounds like {'gte':4} or {'gte':4,'lte':5}
        extra_filter_clauses.append(_score_filter_clause(meta["rating"], f.score))

    must_clauses = _must_clauses_from_query(query, f.text, prof.query_term_map, prof.synonyms)

    # 3) Retrieve vectors (adapter must accept must_clauses & extra_filter_clauses)
    from .pipeline.retrieve import vector_only_retrieve
    ids, X = vector_only_retrieve(
        store,
        size=min_reviews,
        filters=term_filters,
        range_filters=range_filters,
        extra_filter_clauses=extra_filter_clauses,
        must_clauses=must_clauses,
    )
    if len(ids) < 20:
        return (f"Not enough reviews ({len(ids)}) with the specified filters.",
                {"intent": intent_res["intent"], "entities": ents, "meta": meta, "total_vectors": len(ids), "clusters": []})

    score_field = f.score.name  # from config
    probe = store.mget(ids[:20], fields=[score_field])
    bad = [(i, probe.get(i, {}).get(score_field)) for i in ids[:20]
        if probe.get(i, {}).get(score_field) not in (5, "5")]
    print("[DEBUG SCORE CHECK]", bad[:5])  # should be []

    # 4) UMAP/HDBSCAN + exemplars + labels (unchanged) ...
    from .pipeline.reduce import umap_reduce
    from .pipeline.cluster import hdbscan_cluster
    from .pipeline.label import select_exemplars, tfidf_labels

    Z = umap_reduce(X, **(umap_cfg or {}))
    labels, probs, _ = hdbscan_cluster(Z, **(hdbscan_cfg or {}))

    uniq = sorted([c for c in set(labels.tolist()) if c != -1])
    # Compute sizes
    cluster_sizes = {int(c): int((labels == c).sum()) for c in uniq}

    # Debug: print sizes
    print("[DEBUG CLUSTER SIZES]", cluster_sizes)

    if not uniq:
        return ("No significant trends detected by clustering.",
                {"intent": intent_res["intent"], "entities": ents, "meta": meta, "total_vectors": len(ids), "clusters": []})

    exemplars_idx = select_exemplars(Z, labels, k=1)
    rep_ids: List[str] = []
    for c in uniq:
        rep_ids.extend([ids[i] for i in exemplars_idx.get(c, [])])

    docs_map = store.mget(rep_ids, fields=f.text)   # fetch whichever text fields you configured
    # choose the first available text field when building strings
    def _first_text(doc: dict) -> str:
        for tf in f.text:
            if doc.get(tf):
                return doc[tf]
        return ""

    rep_texts = [_first_text(docs_map[_id]) for _id in rep_ids if _id in docs_map]
    cluster_docs = {c: [_first_text(docs_map[ids[i]]) for i in exemplars_idx[c] if ids[i] in docs_map] for c in uniq}
    label_map = tfidf_labels(cluster_docs, top_terms=5)

    # --- start summary block ---
    if summarizer:
        summary_text = summarizer(rep_texts)
    else:
        lines = []
        for c in uniq:
            size = int((labels == c).sum())
            label = label_map.get(c, "(no text)")
            lines.append(f"- Cluster {c} (n={size}): {label}")
        summary_text = "\n".join(lines)

    # 2) optional: get titles from summarizer output (one bullet per cluster, same order)
    # titles = None
    # if summarizer:
    #     raw = summarizer(rep_texts)  # LLM returns bullets
    #     # extract titles between **...** for each bullet, preserving order
    #     titles = []
    #     for line in (raw or "").splitlines():
    #         line = line.strip()
    #         if line.startswith("•") and "**" in line:
    #             try:
    #                 # take the first bold span as the title
    #                 t = line.split("**", 2)[1].strip()
    #                 titles.append(t)
    #             except Exception:
    #                 pass  # be tolerant of odd bullets

    # # 3) compose final lines
    # lines = []
    # for i, c in enumerate(uniq):
    #     size = cluster_sizes[int(c)]
    #     tfidf = label_map.get(c, "")
    #     # prefer LLM title (if we have it), else TF-IDF topic, else generic
    #     title = titles[i] if (titles and i < len(titles)) else (tfidf.title() or f"Trend {i+1}")
    #     lines.append(f"- **{title}** (n={size}) — {tfidf}")

    # summary_text = "\n".join(lines)
    # --- end summary block ---
    details = {
        "intent": intent_res["intent"],
        "intent_scores": intent_res["scores"],
        "entities": ents,
        "meta": meta,
        "profile": profile,
        "index": prof.index,
        "fields": {
            "vector": f.vector, "text": f.text, "timestamp": f.timestamp, "score": f.score.name
        },
        "total_vectors": len(ids),
        "noise_points": int((labels == -1).sum()),
        "clusters": [
            {
                "cluster_id": int(c),
                "size": int((labels == c).sum()),
                "label": label_map.get(c, ""),
                "exemplar_ids": [ids[i] for i in exemplars_idx[c]],
            } for c in uniq
        ],
    }
    return summary_text, details

# def analyze_query_and_summarize(
#     query: str,
#     store: OpenSearchStore,
#     *,
#     min_reviews: int = 2000,
#     text_field: str = "text",
#     umap_cfg: Dict[str, Any] | None = None,
#     hdbscan_cfg: Dict[str, Any] | None = None,
#     summarizer: Optional[Callable[[List[str]], str]] = None,  # optional LLM hook
# ) -> Tuple[str, Dict[str, Any]]:
#     """
#     Returns (summary_text, details_dict).
#     If summarizer is None, returns a simple TF-IDF label list as the summary text.
#     """

#     # 1) intent + entities (your trained models)
#     intent_res = get_intent().predict(query)
#     if intent_res["intent"] != "trend_analysis":
#         return ("I can only analyze trends right now. Please ask about 'top trends'.",
#                 {"intent": intent_res["intent"], "clusters": []})

#     ents = get_ner().extract(query)
#     meta = build_metadata_from_entities(query, ents)

#     if "start_date" not in meta or "end_date" not in meta:
#         return ("Could not determine a date range from your query.",
#                 {"intent": intent_res["intent"], "entities": ents, "meta": meta, "clusters": []})

#     # 2) vector-only retrieval (build OS filters/range)
#     range_filters = {"timestamp": {"gte": meta["start_date"], "lte": meta["end_date"]}}
#     # term_filters  = {}  # add if you derive brand/product etc. from entities
#     term_filters = _term_filters_from_query(query)

#     if meta.get("rating"):
#         range_filters["rating"] = meta["rating"]  # e.g., {'gte': 4} or {'gte':4,'lte':5}

#     ids, X = vector_only_retrieve(
#         store,
#         size=min_reviews,
#         filters=term_filters,
#         range_filters=range_filters
#     )
#     if len(ids) < 20:
#         return (f"Not enough reviews ({len(ids)}) in the specified period to analyze trends.",
#                 {"intent": intent_res["intent"], "entities": ents, "meta": meta, "total_vectors": len(ids), "clusters": []})

#     # 3) UMAP + HDBSCAN
#     Z = umap_reduce(X, **(umap_cfg or {}))
#     labels, probs, _ = hdbscan_cluster(Z, **(hdbscan_cfg or {}))

#     uniq = sorted([c for c in set(labels.tolist()) if c != -1])
#     if not uniq:
#         return ("No significant trends detected by clustering.",
#                 {"intent": intent_res["intent"], "entities": ents, "meta": meta, "total_vectors": len(ids), "clusters": []})

#     # exemplars per cluster
#     exemplars_idx = select_exemplars(Z, labels, k=1)  # one representative per cluster
#     rep_ids: List[str] = []
#     for c in uniq:
#         rep_ids.extend([ids[i] for i in exemplars_idx.get(c, [])])

#     # minimal mget (vector->text)
#     docs_map = store.mget(rep_ids, fields=[text_field])
#     rep_texts = [docs_map[_id].get(text_field, "") for _id in rep_ids if _id in docs_map]

#     # labels for clusters (fast, non-LLM)
#     cluster_docs = {c: [docs_map[ids[i]].get(text_field, "") for i in exemplars_idx[c] if ids[i] in docs_map] for c in uniq}
#     label_map = tfidf_labels(cluster_docs, top_terms=5)

#     # 4) summary
#     if summarizer:
#         # give the summarizer the exemplar texts (one per cluster)
#         summary_text = summarizer(rep_texts)
#     else:
#         # fallback: build a simple human-readable summary from TF-IDF labels
#         lines = []
#         for c in uniq:
#             size = int((labels == c).sum())
#             label = label_map.get(c, "(no text)")
#             lines.append(f"- Cluster {c} (n={size}): {label}")
#         summary_text = "\n".join(lines)

#     details = {
#         "intent": intent_res["intent"],
#         "intent_scores": intent_res["scores"],
#         "entities": ents,
#         "meta": meta,
#         "total_vectors": len(ids),
#         "noise_points": int((labels == -1).sum()),
#         "clusters": [
#             {
#                 "cluster_id": int(c),
#                 "size": int((labels == c).sum()),
#                 "label": label_map.get(c, ""),
#                 "exemplar_ids": [ids[i] for i in exemplars_idx[c]],
#             } for c in uniq
#         ],
#     }
#     return summary_text, details
