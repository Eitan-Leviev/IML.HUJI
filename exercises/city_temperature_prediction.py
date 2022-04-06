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

    df = pd.read_csv(filename)

    # return values

    processed_sample_matrix = df
    response = processed_sample_matrix.Temp
    processed_sample_matrix.drop("Temp", axis=1, inplace=True)

    # TODO should add intercept ?

    # is there nan : print(house_data[house_data["waterfront"] == np.nan].empty)
    # print( house_data.shape)
    # print( house_data.count() )
    # select columns :print(house_data[["condition", "view"]].describe())
    # corr : print( house_data.corr()['price'].sort_values() )

    return processed_sample_matrix, response


if __name__ == '__main__':

    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()