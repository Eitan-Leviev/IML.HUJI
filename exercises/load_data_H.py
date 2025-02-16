from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.metrics import mean_square_error

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


def load_data(filename_samples: str, filename_response: str):

    # pd settings

    pd.set_option("display.max_row", None)

    # read file into dataframe

    samples_data = pd.read_csv(filename_samples)
    response_data = pd.read_csv(filename_response)

    # delete redundant features

    redundant_features = [" Form Name", " Hospital"] # TODO continue
    samples_data.drop(redundant_features, axis=1, inplace=True)

    # value validation:

    # positive features:
    positive_features = ["אבחנה-Age"]
    # u = samples_data["אבחנה-Age"]
    # print(samples_data.describe())
    for feature in positive_features:
        # calculate reliable mean (without garbage values)
        feature_mean = samples_data[samples_data[feature] > 0][feature].mean()
        # replace garbage values with the calculated mean
        samples_data[feature] = np.where(samples_data[feature] > 0, samples_data[feature], feature_mean)
        # print(samples_data["אבחנה-Age"].value_counts())
    # house_data["yr_renovated"] = np.where(house_data["yr_renovated"] > 0, house_data["yr_renovated"], 0)

    # non-negative features:
    # for feature in ["sqft_basement", "sqft_above"]:
    #     # calculate reliable mean (without garbage values)
    #     feature_mean = house_data[house_data[feature] >= 0][feature].mean()
    #     # replace garbage values with the calculated mean
    #     house_data[feature] = np.where(house_data[feature] >= 0, house_data[feature], feature_mean)

    # numeric features:
    for feature in ["lat", "long"]:
        # calculate mean
        feature_mean = house_data[feature].mean()
        # replace nan values with the calculated mean
        house_data[feature].replace(np.nan, feature_mean, inplace=True)

    # features of specific range:
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

    return processed_sample_matrix, response



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

    for feature in X:
        corr = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {feature} Values and Response <br>Pearson Correlation {corr}",
                         labels={"x": f"{feature} Values", "y": "Response Values"})
        fig.write_image(output_path + "/pearson.correlation.%s.png" % feature)

if __name__ == '__main__':

    np.random.seed(0)

    df1 = pd.DataFrame(
        {
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    df4 = pd.DataFrame(
        {
            "E": ["B2", "B3", "B6", "B7"],
            "F": ["D2", "D3", "D6", "D7"],
            "G": ["F2", "F3", "F6", "F7"],
        }
    )

    result = pd.concat([df1, df4], axis=1)

    # Question 1 - Load and preprocessing of housing prices dataset

    X, y = load_data("train.feats.csv",
                     "train.labels.0.csv")

    # Feature evaluation with respect to response
    # feature_evaluation(X, y, "../exercises/house price corr")

    # split train - validation sets
    train_X, train_y, test_X, test_y = split_train_test(X, y)

