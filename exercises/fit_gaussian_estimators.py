from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import math

# constants
mu = 10
sigma = 1
m = 1000

#
uvg = UnivariateGaussian()

def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model

    X = np.random.normal(mu, sigma, size=m) # draw samples
    uvg.fit(X) # fit
    print(f"({uvg.mu_}, {uvg.var_})") # print parameters

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100).astype(np.int) # sample size intervals
    expectation_error_list = []

    for x in ms:
        X = np.random.normal(mu, sigma, size=x)  # draw x samples
        uvg.fit(X)  # fit
        expectation_error_list.append(abs(uvg.mu_ - mu)) # calculate the expectation error of the estimator upon x samples

    #plot
    go.Figure([go.Scatter(x=ms, y=expectation_error_list, mode='lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between Estimated And Real Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$x\\text{ - number of samples}$",
                               yaxis_title="absolute distance between estimated and real expectation",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    # Question 5 - Likelihood evaluation

    # Question 6 - Maximum likelihood
    pass


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
