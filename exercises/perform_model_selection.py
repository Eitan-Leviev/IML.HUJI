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
        .update_layout(title=f"noise {noise} : The true polynomial model and the train + test sets",
                       width=1100,
                       xaxis_title="x",
                       yaxis_title="y")

    fig.write_image(f"../exercises/model selection ex5/part1_noise_{noise}.1.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    train_errors, validation_errors = [], []
    degrees = list(range(11))
    for d in degrees:
        avrg_k_train_error, avrg_k_validation_error = cross_validate(PolynomialFitting(d), train_X.squeeze(), train_y.squeeze(), mean_square_error)
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
        .update_layout(title=f"noise {noise} : Train & validation errors using 5-fold CV for each degree k",
                       width=1100,
                       xaxis_title="k",
                       yaxis_title="error")

    fig.write_image(f"../exercises/model selection ex5/part1_noise_{noise}.2.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    poly_model = PolynomialFitting(selected_k)
    poly_model.fit(np.array(train_X).flatten(), np.array(train_y).flatten())
    y_pred = poly_model.predict(np.array(test_X).flatten())
    test_error = mean_square_error(np.array(test_y), y_pred)
    print(f"noise {noise} selected k: {selected_k}")
    print(f"validation error: {np.round(selected_error, 2)}\ntest error: {np.round(test_error, 2)}")


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

    X, y = datasets.load_diabetes(return_X_y=True) # X: (442, 10). y: (442,).   as np.array
    x_train = X[:n_samples]
    y_train = y[:n_samples]
    x_test = X[n_samples:]
    y_test = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    lambdas = 10 ** np.linspace(-3, 0.5, 100)

    ridge_train_errors, ridge_validation_errors = [], []
    lasso_train_errors, lasso_validation_errors = [], []

    for lam in lambdas:
        # CV for ridge
        ridge_model = RidgeRegression(lam)
        ridge_model.fit(x_train, y_train)
        ridge_train_score, ridge_validation_score = cross_validate(ridge_model, x_train, y_train, mean_square_error)
        ridge_train_errors.append(ridge_train_score)
        ridge_validation_errors.append(ridge_validation_score)
        # print(ridge_train_score)
        # print(ridge_validation_score)

        # CV for lasso
        lasso_model = Lasso(alpha=lam, normalize=True, max_iter=10000, tol=1e-4)
        lasso_train_score, lasso_validation_score = cross_validate(lasso_model, x_train, y_train, mean_square_error)
        lasso_train_errors.append(lasso_train_score)
        lasso_validation_errors.append(lasso_validation_score)
        # print(lasso_train_score)
        # print(lasso_validation_score)

    ridge_min_ind = np.argmin(np.array(ridge_validation_errors))
    ridge_selected_lam = np.array(lambdas)[ridge_min_ind]
    selected_error = ridge_validation_errors[ridge_min_ind]

    fig = go.Figure([
        go.Scatter(name='train errors', x=lambdas, y=ridge_train_errors, mode='markers', marker_color='blue'),
        go.Scatter(name='validation errors', x=lambdas, y=ridge_validation_errors, mode='markers'),
        go.Scatter(name='Selected lam', x=[ridge_selected_lam], y=[selected_error], mode='markers', marker=dict(color='darkred', symbol="x", size=10),
                   marker_color='green')
    ]) \
        .update_layout(title=f"Ridge estimator: train and validation errors as func of regularization parameter",
                       width=1100,
                       xaxis_title="lambda",
                       yaxis_title="mse")

    fig.write_image(f"../exercises/model selection ex5/q7.1.png")

    lasso_min_ind = np.argmin(np.array(lasso_validation_errors))
    lasso_selected_lam = np.array(lambdas)[lasso_min_ind]
    selected_error = lasso_validation_errors[lasso_min_ind]

    fig = go.Figure([
        go.Scatter(name='train errors', x=lambdas, y=lasso_train_errors, mode='markers', marker_color='blue'),
        go.Scatter(name='validation errors', x=lambdas, y=lasso_validation_errors, mode='markers', marker_color='green'),
        go.Scatter(name='Selected lam', x=[lasso_selected_lam], y=[selected_error], mode='markers', marker=dict(color='darkred', symbol="x", size=10))
    ]) \
        .update_layout(title=f"Lasso estimator: train and validation errors as func of regularization parameter",
                       width=1100,
                       xaxis_title="lambda",
                       yaxis_title="mse")

    fig.write_image(f"../exercises/model selection ex5/q7.2.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model

    # fit linear regression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(f"linear regression loss: {lr.loss(x_test, y_test)}")

    # fit ridge
    print(ridge_selected_lam)
    ridge_model = RidgeRegression(ridge_selected_lam)
    ridge_model.fit(x_train, y_train)
    print(f"ridge loss: {ridge_model.loss(x_test, y_test)}")

    # fit lasso
    print(lasso_selected_lam)
    lasso_model = Lasso(alpha=lasso_selected_lam, normalize=True, max_iter=10000, tol=1e-4)
    lasso_model.fit(x_train, y_train)
    print(f"lasso loss: {mean_square_error(y_test, lasso_model.predict(x_test))}")

if __name__ == '__main__':

    np.random.seed(0)

    select_polynomial_degree()

    # Question 4
    select_polynomial_degree(noise=0)

    # Question 5
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
