import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    # read file into dataframe
    data = pd.read_csv(filename, parse_dates=True)

    # value validation
    data = data[data["Temp"] > -50]
    data = data[data["Temp"] < 80]

    # new features
    data['DayOfYear'] = pd.to_datetime(data['Date']).dt.dayofyear

    # return values
    processed_sample_matrix = data
    response = processed_sample_matrix.Temp
    processed_sample_matrix.drop("Temp", axis=1, inplace=True)

    # TODO should add intercept ?

    return processed_sample_matrix, response


if __name__ == '__main__':

    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country

    # Question 3 - Exploring differences between countries

    # Question 4 - Fitting model for different values of `k`

    # Question 5 - Evaluating fitted model on different countries
