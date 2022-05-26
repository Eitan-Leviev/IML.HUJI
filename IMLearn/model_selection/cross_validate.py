from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    train_errors, validation_errors = [], []
    folds = list(range(cv))
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    for i in folds:
        # split
        small_training_X = np.concatenate(X_folds[:i] + X_folds[i+1:])
        small_training_y = np.concatenate(y_folds[:i] + y_folds[i+1:])
        validation_X = X_folds[i]
        validation_y = y_folds[i]
        # fit model with i'th small training
        estimator.fit(small_training_X, small_training_y)
        # predict for training
        training_pred = estimator.predict(small_training_X)
        # predict for validation
        validation_pred = estimator.predict(validation_X)
        # add small-training score to list
        training_score = scoring(small_training_y, training_pred)
        train_errors.append(training_score)
        # add validation score to list
        validation_score = scoring(validation_y, validation_pred)
        validation_errors.append(validation_score)

    # calc the avrg small-training errors
    avrg_training_error = np.array(train_errors).mean()
    # calc the avrg validation errors
    avrg_validation_error = np.array(validation_errors).mean()

    return avrg_training_error, avrg_validation_error
