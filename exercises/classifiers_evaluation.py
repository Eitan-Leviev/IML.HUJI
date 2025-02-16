import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

import sys
sys.path.append("../")


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"), ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        X, y_true = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def my_callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(X, y_true))

        Perceptron(callback=my_callback).fit(X, y_true)

        fig = go.Figure([go.Scatter(x=list(range(len(losses))), y=losses, mode='lines')],
                  layout=go.Layout(
                      title="loss as a function of perceptron's iteration",
                      xaxis_title="current iteration",
                      yaxis_title="loss",
                      height=600,
                  width=1200))
        fig.write_image(f"../exercises\classifiers_evaluation/{n}_loss.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for n, f in [("gaussian1", "../datasets/gaussian1.npy"), ("gaussian2", "../datasets/gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over train set
        nbe = GaussianNaiveBayes()
        ldae = LDA()
        nbe.fit(X, y), ldae.fit(X, y)
        predictions = (nbe.predict(X), ldae.predict(X))

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        from IMLearn.metrics import accuracy
        symbols = np.array(["cross", "circle"])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'gaussian naive bayes with accuracy<br> of {accuracy(y, predictions[0])}',
                                            f'LDA with accuracy <br>of {accuracy(y, predictions[1])}'),
                            horizontal_spacing=0.05)

        # Add traces for data-points setting symbols and colors
        for i, y_pred in enumerate(predictions):
            comp = [1 if y_pred[j] == y[j] else 0 for j in range(len(y_pred))]
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y_pred,
                                                   symbol=symbols[comp],
                                                   line=dict(color="black", width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=np.array(nbe.mu_)[:, 0], y=np.array(nbe.mu_)[:, 1], mode="markers",
                                 showlegend=False,
                                 marker=dict(color="blue", symbol="x", size=20, line=dict(color="black", width=1))), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.array(ldae.mu_)[:, 0], y=np.array(ldae.mu_)[:, 1], mode="markers",
                                 showlegend=False,
                                 marker=dict(color="blue", symbol="x", size=20, line=dict(color="black", width=1))), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_traces([
            get_ellipse(np.array(nbe.mu_)[0, :], np.diag(np.array(nbe.vars_)[0,:])),
            get_ellipse(np.array(nbe.mu_)[1, :], np.diag(np.array(nbe.vars_)[1, :])),
            get_ellipse(np.array(nbe.mu_)[2, :], np.diag(np.array(nbe.vars_)[2, :]))
        ], rows=1, cols=1)

        fig.add_traces([
            get_ellipse(np.array(ldae.mu_)[0, :], ldae.cov_),
            get_ellipse(np.array(ldae.mu_)[1, :], ldae.cov_),
            get_ellipse(np.array(ldae.mu_)[2, :], ldae.cov_)
        ], rows=1, cols=2)

        fig.write_image(f"../exercises\classifiers_evaluation/{n}.png")

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
