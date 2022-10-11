import numpy as np


def save(save_path, D_w: np.ndarray):
    np.save(save_path, D_w)
    print(f"save to {save_path}.")


def load(load_path):
    if load_path is None:
        return None
    try:
        return np.load(load_path)
    except ValueError:
        raise ValueError("No such a file.")
