import umap
import numpy as np

def umap_reduce(X: np.ndarray, *, n_components=20, n_neighbors=30, min_dist=0.0,
                metric="cosine", random_state=42):
    n = X.shape[0]
    # adapt n_neighbors to dataset size
    nn = max(2, min(n_neighbors, max(2, n - 1)))
    reducer = umap.UMAP(
        n_components=min(n_components, max(2, n)),  # cap by n too
        n_neighbors=nn,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)
