from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import math

def test_univariate_gaussian():

    mu = 10
    sigma = 1
    m = 1000
    uvg = UnivariateGaussian()

    # Question 1 - Draw samples and print fitted model

    X = np.random.normal(mu, sigma, size=m) # draw samples
    uvg.fit(X) # fit
    print(f"({uvg.mu_}, {uvg.var_})") # print parameters

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100).astype(np.int) # sample size intervals
    expectation_errors = []

    for i in ms:
        uvg.fit(X[:i])  # fit
        expectation_errors.append(abs(uvg.mu_ - mu)) # calculate the expectation error of the estimator upon i samples

    #plot
    # go.Figure([go.Scatter(x=ms, y=expectation_errors, mode='lines')],
    #           layout=go.Layout(title=r"$\text{Absolute Distance Between Estimated And Real Expectation As Function Of Number Of Samples}$",
    #                            xaxis_title="number of samples",
    #                            yaxis_title="absolute distance between estimated and real expectation",
    #                            height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    # go.Figure([go.Scatter(x=X, y=uvg.pdf(X), mode='markers')],
    #           layout=go.Layout(
    #               title=r"$\text{PDF Of The Samples Based On Q1's Model - As Function Of Sample Values}$",
    #               xaxis_title="sample values",
    #               yaxis_title="the PDF of given sample",
    #               height=600)).show()

    # quiz

    # X_ = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #       -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    #
    # print(uvg.fit(X_).mu_)
    # print( uvg.log_likelihood( 1, 1, X_) )
    # print( uvg.log_likelihood( 10, 1, X_) )


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model

    mvg = MultivariateGaussian()
    m = 1000
    mu = np.array([0,0,4,0])
    sigma = np.array([(1, 0.2, 0, 0.5),
                      (0.2, 2, 0, 0),
                      (0, 0, 1, 0),
                      (0.5, 0, 0, 1)],
                     dtype = float)

    X = np.random.multivariate_normal(mu, sigma, size=m)  # draw samples
    mvg.fit(X) # fit
    print("estimated expectation (unbiased estimator):\n", mvg.mu_)
    print("estimated covariance matrix (unbiased estimator):\n", mvg.cov_)

    # test to pdf

    X = np.random.multivariate_normal(mu, sigma, size=3)  # draw samples
    print(X)
    # mvg.pdf(X)
    print("should be 3x1:\n", mvg.pdf(X))

    # Question 5 - Likelihood evaluation

    # Question 6 - Maximum likelihood
    pass


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
