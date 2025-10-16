import hdbscan
import numpy as np
from typing import Tuple

def hdbscan_cluster(Z: np.ndarray, *, min_cluster_size=25, min_samples=None, metric="euclidean") -> Tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    n = Z.shape[0]
    # adapt cluster size for tiny batches
    mcs = max(5, min(min_cluster_size, max(5, n // 10)))  # ~10% of N, floor 5
    model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(Z)
    probs = getattr(model, "probabilities_", None)
    return labels, probs, model
