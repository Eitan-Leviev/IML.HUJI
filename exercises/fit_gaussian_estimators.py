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
    print(f"({uvg.mu_}, {uvg.var_})\n") # print parameters

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.linspace(10, 1000, 100).astype(int) # sample size intervals
    expectation_errors = []

    for i in ms:
        uvg.fit(X[:i])  # fit
        expectation_errors.append(abs(uvg.mu_ - mu)) # calculate the expectation error of the estimator upon i samples

    #plot
    go.Figure([go.Scatter(x=ms, y=expectation_errors, mode='lines')],
              layout=go.Layout(title=r"$\text{Absolute Distance Between Estimated And Real Expectation As Function Of Number Of Samples}$",
                               xaxis_title="number of samples",
                               yaxis_title="absolute distance between estimated and real expectation",
                               height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    go.Figure([go.Scatter(x=X, y=uvg.pdf(X), mode='markers')],
              layout=go.Layout(
                  title=r"$\text{PDF Of The Samples Based On Q1's Model - As Function Of Sample Values}$",
                  xaxis_title="sample values",
                  yaxis_title="the PDF of given sample",
                  height=600)).show()


def test_multivariate_gaussian():

    mvg = MultivariateGaussian()
    m = 1000
    mu = np.array([0,0,4,0])
    sigma = np.array([(1, 0.2, 0, 0.5),
                      (0.2, 2, 0, 0),
                      (0, 0, 1, 0),
                      (0.5, 0, 0, 1)],
                     dtype = float)

    # Question 4 - Draw samples and print fitted model

    X = np.random.multivariate_normal(mu, sigma, size=m)  # draw m samples
    mvg.fit(X) # fit
    print("estimated expectation (unbiased estimator):\n", mvg.mu_, "\n")
    print("estimated covariance matrix (unbiased estimator):\n", mvg.cov_, "\n")

    # Question 5 - Likelihood evaluation

    ms = np.linspace(-10, 10, 200)

    # building X_ Y_ arrays :
    length = len(ms)
    x_ = np.array([ms])
    ms = np.array([ms])
    #
    for i in range(length - 1):
        x_ = np.concatenate((x_, ms), axis=0)
    #
    y_ = x_.T.ravel()
    x_ = x_.ravel()

    # setup for Q6
    max_likelihood = float('-inf')
    argmax = (0, 0)

    # building Z array + finding max-likelihood & argmax of it :
    z = []
    for f3, f1 in zip(x_, y_):
        curr_mu = np.array([f1, 0, f3, 0])
        curr_likelihood = mvg.log_likelihood(curr_mu, sigma, X)
        z.append(curr_likelihood)
        #
        if curr_likelihood > max_likelihood:
            max_likelihood = curr_likelihood
            argmax = (f1, f3)

    # plot
    go.Figure(go.Heatmap(x=x_, y=y_, z=z), layout=go.Layout(
        title="heatmap of likelihood as a function of f1 f3 - represent the expectation [f1, 0, f3, 0]",
        xaxis_title="f3",
        yaxis_title="f1",
        height=300*2, width=200*4)).show()

    # Question 6 - Maximum likelihood

    print("the maximum log-likelihood: ", round(max_likelihood, 3), "\n")
    print(f"the argmax (f1, f3): ({round(argmax[0], 3)}, {round(argmax[1], 3)}) \n")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
