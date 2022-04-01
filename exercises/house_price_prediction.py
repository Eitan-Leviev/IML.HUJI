from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import sys
sys.path.append("../")
from utils import *

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # read file into dataframe

    house_data = pd.read_csv(filename)

    # delete redundant features

    house_data.drop(["id", "zipcode", "date"], axis=1, inplace=True)

    # value validation

    # positive features:
    for feature in ["price", "bedrooms", "sqft_living", "sqft_lot", "yr_built", "sqft_living15", "sqft_lot15", "bathrooms", "floors"]:
        # calculate reliable mean (without garbage values)
        feature_mean = house_data[house_data[feature] > 0][feature].mean()
        # replace garbage values with the calculated mean
        house_data[feature] = np.where(house_data[feature] > 0, house_data[feature], feature_mean)
    house_data["yr_renovated"] = np.where(house_data["yr_renovated"] > 0, house_data["yr_renovated"], 0)

    # non-negative features:
    for feature in ["sqft_basement", "sqft_above"]:
        # calculate reliable mean (without garbage values)
        feature_mean = house_data[house_data[feature] >= 0][feature].mean()
        # replace garbage values with the calculated mean
        house_data[feature] = np.where(house_data[feature] >= 0, house_data[feature], feature_mean)

    # TODO are lat long helps ?
    # numeric features:
    for feature in ["lat", "long"]:
        # calculate mean
        feature_mean = house_data[feature].mean()
        # replace nan values with the calculated mean
        house_data[feature].replace(np.nan, feature_mean, inplace=True)

    # features of specific range:
    # TODO does rounding ( round(x) ) helps ?
    # calculate reliable mean (without garbage values)
    feature_mean = house_data[house_data["waterfront"].isin([0, 1])]["waterfront"].mean()
    # replace garbage values with the calculated mean
    house_data["waterfront"] = np.where(house_data["waterfront"].isin([0, 1]), house_data["waterfront"], feature_mean)
    #
    # calculate reliable mean (without garbage values)
    feature_mean = house_data[house_data["view"].isin(range(5))]["view"].mean()
    # replace garbage values with the calculated mean
    house_data["view"] = np.where(house_data["view"].isin(range(5)), house_data["view"], feature_mean)
    #
    # calculate reliable mean (without garbage values)
    feature_mean = house_data[house_data["condition"].isin(range(1, 6))]["condition"].mean()
    # replace garbage values with the calculated mean
    house_data["condition"] = np.where(house_data["condition"].isin(range(1, 6)), house_data["condition"], feature_mean)
    #
    # calculate reliable mean (without garbage values)
    feature_mean = house_data[house_data["grade"].isin(range(1, 15))]["grade"].mean()
    # replace garbage values with the calculated mean
    house_data["grade"] = np.where(house_data["grade"].isin(range(1, 15)), house_data["grade"], feature_mean)

    # new features

    # decade_renovated:
    house_data["decade_renovated"] = (house_data["yr_renovated"] / 10).astype(int)
    house_data.drop(["yr_renovated"], axis=1, inplace=True)
    # create dummies. if decade_renovated contains 0 : do not create dummy for the 0'th decade
    if not house_data[house_data["decade_renovated"] == 0].empty:
        house_data = pd.get_dummies(house_data, prefix='decade_renovated_', columns=['decade_renovated'], drop_first=True)
    else:
        house_data = pd.get_dummies(house_data, prefix='decade_renovated_', columns=['decade_renovated'])
    # decade_built:
    house_data["decade_built"] = (house_data["yr_built"] / 10).astype(int)
    house_data.drop(["yr_built"], axis=1, inplace=True)
    house_data = pd.get_dummies(house_data, prefix='decade_built_', columns=['decade_built'])

    # return values

    processed_sample_matrix = house_data
    response = processed_sample_matrix.price
    processed_sample_matrix.drop("price", axis=1, inplace=True)

    # TODO should add intercept ?

    return processed_sample_matrix, response

    # is there nan : print(house_data[house_data["waterfront"] == np.nan].empty)
    # print( house_data.shape)
    # print( house_data.count() )
    # select columns :print(house_data[["condition", "view"]].describe())
    # corr : print( house_data.corr()['price'].sort_values() )


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    load_data("../datasets/house_prices.csv")
    # load_data("house_prices.csv")
    # load_data("../house_prices.csv")
    # load_data("house_prices")
    # load_data("../IML.HUJI/datasets/house_prices.csv")
    # load_data("C:\Users\eitan\IML.HUJI\datasets\house_prices.csv")
    # load_data("C:/Users/eitan/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response

    # Question 3 - Split samples into training- and testing sets.

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
