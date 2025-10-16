# trendminer/pipeline/label.py
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def select_exemplars(Z: np.ndarray, labels: np.ndarray, k: int = 5) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for c in np.unique(labels):
        if c == -1: 
            continue
        idx = np.where(labels == c)[0]
        centroid = Z[idx].mean(axis=0, keepdims=True)
        d = np.linalg.norm(Z[idx] - centroid, axis=1)
        out[int(c)] = idx[np.argsort(d)[:k]].tolist()
    return out

def _clean_docs(docs: List[str]) -> List[str]:
    cleaned = []
    for d in docs:
        if not d:
            continue
        t = d.strip()
        # discard ultra-short or whitespace-only docs
        if len(t) < 5:
            continue
        cleaned.append(t)
    return cleaned

def tfidf_labels(cluster_docs: Dict[int, List[str]], top_terms: int = 5) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for c, docs in cluster_docs.items():
        docs = docs[:50] if docs else []
        docs = _clean_docs(docs)

        if not docs:
            labels[c] = "(no text)"
            continue

        try:
            # Use a looser token pattern so we don't drop short tokens/numbers
            vec = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                stop_words="english",
                token_pattern=r"(?u)\b\w+\b",   # accept 1+ word chars
                min_df=1                        # don’t require frequency threshold
            )
            X = vec.fit_transform(docs)
            if X.shape[1] == 0:
                labels[c] = "(no vocabulary)"
                continue
            terms = vec.get_feature_names_out()
            weights = X.sum(axis=0).A1
            top_idx = np.argsort(-weights)[:max(1, top_terms)]
            top = terms[top_idx]
            labels[c] = ", ".join(top)
        except ValueError:
            # e.g., "empty vocabulary"
            labels[c] = "(no vocabulary)"
    return labels
