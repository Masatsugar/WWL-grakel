import argparse

import numpy as np
from grakel.datasets import fetch_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from wwl import WassersteinWeisfeilerLehman


def cross_validation(K, y, model, cv):
    accuracy_scores = []
    for train_index, test_index in cv.split(K, y):
        K_train = K[train_index][:, train_index]
        K_test = K[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(K_train, y_train)
        y_pred = model.predict(K_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    print(
        "Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %".format(
            np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100
        )
    )


def main(args):
    np.random.seed(42)

    # load MUTAG dataset
    MUTAG = fetch_dataset("MUTAG", verbose=False)
    G, y = MUTAG.data, MUTAG.target

    # compute Wasserstein distance
    wwl = WassersteinWeisfeilerLehman(n_iter=args.n_iter)
    w_dist_mat = wwl.compute_wasserstein_distance(G)
    # save(wwl_kernel, M)

    M = w_dist_mat[2]

    # Cross Validation
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    if args.grid_search:
        gammas = np.logspace(-4, 1, num=6)
        for gamma in gammas:
            # compute laplacian kernel
            K = np.exp(-gamma * M)
            Cs = np.logspace(-3, 3, num=7)
            for C in Cs:
                print(f"C={C}, gamma={gamma}")
                cross_validation(K, y, C, cv)
    else:
        C = 1
        gamma = 10
        model = SVC(C=C, kernel="precomputed")
        K = np.exp(-gamma * M)
        cross_validation(K, y, model, cv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv",
        default=False,
        action="store_true",
        help="Enable a 10-fold crossvalidation",
    )
    parser.add_argument(
        "--gridsearch", default=False, action="store_true", help="Enable grid search"
    )
    parser.add_argument(
        "--sinkhorn",
        default=False,
        action="store_true",
        help="Use sinkhorn approximation",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        required=False,
        default=2,
        help="(Max) number of WL iterations",
    )
    args = parser.parse_args()
    main(args)
