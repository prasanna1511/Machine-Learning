import numpy as np
from typing import Optional

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function should calculate the root mean squared error given target y and prediction y_pred

    Args:
        - y(np.array): target data
        - y_pred(np.array): predicted data

    Returns:
        - err (float): root mean squared error between y and y_pred

    """

    rmse = np.sqrt(np.mean((y - y_pred)**2))

    err = rmse

    return err

def split_data(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training, validation

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation

    """
    num_samples = len(x)
    val_samples = int(num_samples * 0.5)

    indices = np.random.permutation(num_samples)
    val_indices = indices[:val_samples]
    train_indices = indices[val_samples:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]


    # x_train = None
    # y_train = None
    # x_val = None
    # y_val = None

    return x_train, y_train, x_val, y_val