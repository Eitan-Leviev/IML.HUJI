from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from typing import Tuple, Callable


from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    x = np.linspace(-1.2, 2, num=n_samples)
    eps = np.random.normal(0, np.sqrt(noise), n_samples)
    f = (x+3) * (x+2) * (x+1) * (x-1) * (x-2)
    y = f + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), train_proportion=2/3)

    # plot
    fig = go.Figure([
        go.Scatter(name='true model', x=x, y=f, mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='train set', x=train_X.squeeze(), y=train_y, mode='markers', marker_color='rgb(25,115,132)'),
        go.Scatter(name='test set', x=test_X.squeeze(), y=test_y, mode='markers', marker_color='rgb(152,171,25)')
                     ]) \
        .update_layout(title="The true polynomial model and the train + test sets",
                       width=1100,
                       xaxis_title="x",
                       yaxis_title="y")

    fig.write_image(f"../exercises/model selection ex5/q1.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    train_errors, validation_errors = [], []
    degrees = list(range(11))
    for d in degrees:
        avrg_k_train_error, avrg_k_validation_error = cross_validate(PolynomialFitting(d), train_X.to_numpy(), train_y.to_numpy(), mean_square_error)
        train_errors.append(avrg_k_train_error)
        validation_errors.append(avrg_k_validation_error)

    min_ind = np.argmin(np.array(validation_errors))
    selected_k = np.array(degrees)[min_ind]
    selected_error = validation_errors[min_ind]

    # plot
    fig = go.Figure([
        go.Scatter(name='train error', x=degrees, y=train_errors, mode='markers+lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='validation error', x=degrees, y=validation_errors, mode='markers+lines', marker_color='rgb(152,171,25)'),
        go.Scatter(name='Selected k', x=[selected_k], y=[selected_error], mode='markers', marker=dict(color='darkred', symbol="x", size=10))
    ]) \
        .update_layout(title="Train & validation errors using 5-fold CV for each degree k",
                       width=1100,
                       xaxis_title="k",
                       yaxis_title="error")

    fig.write_image(f"../exercises/model selection ex5/q2.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    X, y = datasets.load_diabetes(return_X_y=True) # X: (442, 10). y: (442,)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_regularization_parameter()
