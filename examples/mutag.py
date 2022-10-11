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

    dataset = fetch_dataset(args.dataset, verbose=False)
    G, y = dataset.data, dataset.target

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
        gammas = np.logspace(-4, 1, num=10)
        for gamma in gammas:
            # compute laplacian kernel
            K = np.exp(-gamma * M)
            Cs = np.logspace(-3, 3, num=10)
            for C in Cs:
                print(f"C={C}, gamma={gamma}")
                model = SVC(C=C, kernel="precomputed")
                accuracy_scores = cross_validation(K, y, model, cv)
                print(
                    f"WWL(C={C}, gamma={gamma}): Mean {args.cv}-fold accuracy: "
                    f"{np.mean(accuracy_scores) * 100:2.2f} +- {np.std(accuracy_scores) * 100:2.2f} %"
                )
    else:
        C = args.C
        gamma = args.gamma

        model = SVC(C=C, kernel="precomputed")
        K = np.exp(-gamma * M)
        accuracy_scores = cross_validation(K, y, model, cv)
        print(
            f"WWL(C={C}, gamma={gamma}): Mean {args.cv}-fold accuracy: "
            f"{np.mean(accuracy_scores) * 100:2.2f} +- {np.std(accuracy_scores) * 100:2.2f} %"
        )

    model = SVC(C=C, kernel="precomputed")
    K = wwl.wl_kernel.fit_transform(G)
    accuracy_scores = cross_validation(K, y, model, cv)
    print(
        f"WL(C={C}): Mean {args.cv}-fold accuracy: "
        f"{np.mean(accuracy_scores) * 100:2.2f} +- {np.std(accuracy_scores) * 100:2.2f} %"
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
    parser.add_argument(
        "--C",
        type=float,
        required=False,
        default=1.0,
        help="SVM C",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        required=False,
        default=10.0,
        help="Laplacian kernel gamma",
    )
    args = parser.parse_args()
    main(args)
