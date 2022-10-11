import argparse
from pathlib import Path

import numpy as np
from grakel.datasets import fetch_dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import wwl_grakel.utils
from wwl_grakel import WassersteinWeisfeilerLehman


def cross_validation(K, y, model, cv):
    accuracy_scores = []
    for train_index, test_index in cv.split(K, y):
        K_train = K[train_index][:, train_index]
        K_test = K[test_index][:, train_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(K_train, y_train)
        y_pred = model.predict(K_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    return accuracy_scores


def main(args):
    np.random.seed(42)

    # load MUTAG dataset

    MUTAG = fetch_dataset(args.dataset, verbose=False)
    G, y = MUTAG.data, MUTAG.target

    wwl = WassersteinWeisfeilerLehman(n_iter=args.n_iter)
    save_path = Path(f"outputs/wl_{wwl.ground_distance}_embeddings_h{wwl.n_iter}.npy")
    if save_path.exists():
        Ms = wwl_grakel.utils.load(save_path)
    else:
        Ms = wwl.compute_wasserstein_distance(G)
        wwl_grakel.utils.save(save_path, Ms)

    M = 0
    for _M in Ms:
        M += _M

    # Cross Validation
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True)
    if args.gridsearch:
        gammas = np.logspace(-4, 1, num=6)
        for gamma in gammas:
            # compute laplacian kernel
            K = np.exp(-gamma * M)
            Cs = np.logspace(-3, 3, num=7)
            for C in Cs:
                print(f"C={C}, gamma={gamma}")
                model = SVC(C=C, kernel="precomputed")
                accuracy_scores = cross_validation(K, y, model, cv)
                print(
                    "Mean {}-fold accuracy: {:2.2f} +- {:2.2f} %".format(
                        args.cv,
                        np.mean(accuracy_scores) * 100,
                        np.std(accuracy_scores) * 100,
                    )
                )
    else:
        C = 1
        gamma = 10
        model = SVC(C=C, kernel="precomputed")
        K = np.exp(-gamma * M)
        accuracy_scores = cross_validation(K, y, model, cv)

        print(
            "Mean {}-fold accuracy: {:2.2f} +- {:2.2f} %".format(
                args.cv, np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="MUTAG", type=str, help="select from grakel dataset"
    )
    parser.add_argument(
        "--cv",
        default=10,
        type=int,
        help=" k-fold cross validation",
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
