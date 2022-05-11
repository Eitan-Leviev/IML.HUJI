import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):

    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train and test errors of AdaBoost

    test_loss, train_loss = [], []

    adaB = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    adaB.fit(train_X, train_y)
    n = np.arange(start=1, stop=n_learners + 1)

    for t in n:
        train_loss.append(adaB.partial_loss(train_X, train_y, t))
        test_loss.append(adaB.partial_loss(test_X, test_y, t))

    fig = go.Figure([go.Scatter(x=n, y=train_loss,
                   mode='markers + lines', name='train'),
                     go.Scatter(x=n, y=test_loss,
                   mode='markers + lines', name='test')])

    fig.update_layout(
        title=f"loss of adaboost classifier over train & test with decision tree of {noise} noise",
        xaxis=dict(title="learners"),
        yaxis=dict(title="error"))

    fig.write_image(f"../exercises/adaboost_scenario/adaboost_losses.png")

    # Question 2: Plotting decision surfaces

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in T],
                        horizontal_spacing=.03, vertical_spacing=.03)

    for j, t in enumerate(T):

        fig.add_traces(
            [decision_surface(
            lambda X: adaB.partial_predict(X, t), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=(test_y == 1).astype(int),
                                   symbol=class_symbols[test_y.astype(int)],
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=0.5)))],
            rows=(j // 2) + 1, cols=(j % 2) + 1)

    fig.update_layout(
        title=f"decision boundary of adaboost with decision tree of {noise} noise as a func of learners num",
        width=800, height=800, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)

    fig.write_image(f"../exercises/adaboost_scenario/decision_boundary.png")

    # Question 3: Decision surface of best performing ensemble

    from IMLearn.metrics import accuracy

    best = np.argmin(test_loss) + 1

    fig = go.Figure(data=[decision_surface(
        lambda X: adaB.partial_predict(X, best), lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                   mode="markers", showlegend=False, marker=dict(color=(test_y == 1).astype(int),
                               symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=0.5)))])

    the_accuracy = accuracy(test_y, adaB.partial_predict(test_X, best))
    fig.update_layout(
        title=f"decision surface of best committee of adaboost with decision tree of {noise} noise,"
              f"<br> committee size={best} and Accuracy={the_accuracy}",
        width=800, height=800, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)

    fig.write_image(f"../exercises/adaboost_scenario/decision_surface.png")

    # Question 4: Decision surface with weighted samples

    fig = go.Figure(data=[decision_surface(
        adaB.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:,0], y=train_X[:,1], mode="markers", showlegend=False,
                   marker=dict(color=(train_y == 1).astype(int),
                               symbol=class_symbols[train_y.astype(int)],
                               colorscale=[custom[0],custom[-1]],
                               line=dict(color="black", width=1),
                               size=(adaB.D_ / np.max(adaB.D_)) * 5))])

    fig.update_layout(
        title=f"decision surface of weighted samples adaboost with decision tree of {noise} noise",
        width=800, height=800, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)

    fig.write_image(f"../exercises/adaboost_scenario/weighted.png")

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    # fit_and_evaluate_adaboost(0.4)
