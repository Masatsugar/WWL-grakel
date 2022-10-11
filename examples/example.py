from pathlib import Path

import numpy as np
from grakel.datasets import fetch_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import wwl_grakel
from wwl_grakel import WassersteinWeisfeilerLehman

# Contains accuracy scores for each cross validation step; the
# means of this list will be used later on.


if __name__ == "__main__":
    np.random.seed(42)

    MUTAG = fetch_dataset("MUTAG", verbose=False)
    G, y = MUTAG.data, MUTAG.target

    # Compute Wasserstein distance
    wwl = WassersteinWeisfeilerLehman(n_iter=2)

    save_path = Path(f"outputs/wl_{wwl.ground_distance}_embeddings_h{wwl.n_iter}.npy")
    if save_path.exists():
        M = wwl_grakel.utils.load(save_path)
    else:
        M = wwl.compute_wasserstein_distance(G)
        wwl_grakel.utils.save(save_path, M)

    # compute Laplacian kernel
    K = np.exp(-0.001 * M)

    K_train, K_test, y_train, y_test = train_test_split(K, y, test_size=0.1)

    model = SVC(C=100, kernel="precomputed")
    model.fit(K_train, y_train)
    y_pred = model.predict(K_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
