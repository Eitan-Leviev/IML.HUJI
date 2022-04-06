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

    house_data.drop(["id", "date", "zipcode"], axis=1, inplace=True)

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

    # is there nan : print(house_data[house_data["waterfront"] == np.nan].empty)
    # print( house_data.shape)
    # print( house_data.count() )
    # select columns :print(house_data[["condition", "view"]].describe())
    # corr : print( house_data.corr()['price'].sort_values() )

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

    # TODO what about intercept ?

    # TODO uncomment:

    # for feature in X:
    #     corr = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
    #     fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
    #                      title=f"Correlation Between {feature} Values and Response <br>Pearson Correlation {corr}",
    #                      labels={"x": f"{feature} Values", "y": "Response Values"})
    #     fig.write_image(output_path + "/pearson.correlation.%s.png" % feature)

def question_4():

    lr = LinearRegression(True)

    # features_num = len(train_y)

    mse_list = []
    var_loss = []

    for p in range(10, 101):

        p_loss_list = []

        for _ in range(10):
            # 1) Sample p% of the overall training data
            rand_p_sample = train_X.sample(frac=p / 100)
            p_response = train_y.reindex_like(rand_p_sample)
            # 2) Fit linear model (including intercept) over sampled set
            lr.fit(rand_p_sample, p_response)
            # 3) Test fitted model over test set
            p_loss_tmp = lr.loss(test_X, test_y)
            p_loss_list.append(p_loss_tmp)

        # 4) Store average and variance of loss over test set
        if len(p_loss_list) == 0: print("division by zero"), exit(1)  # no reason to happen but just to ensure.
        p_loss_avrg = sum(p_loss_list) / len(p_loss_list)
        p_loss_var = np.array(p_loss_list).var()  # TODO to be tested
        mse_list.append(p_loss_avrg)
        var_loss.append(p_loss_var)

    std_loss = np.sqrt(np.array(var_loss))
    mse = np.array(mse_list)
    ms = np.array(list(range(10, 101)))

    fig = go.Figure([go.Scatter(x=ms, y=mse - 2 * std_loss, fill=None, mode="lines", line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=ms, y=mse + 2 * std_loss, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False),
                     go.Scatter(x=ms, y=mse, mode="markers+lines", marker=dict(color="black", size=1),
                                showlegend=False)],
                    layout=go.Layout(title="Model Evaluation Over Increasing Portions Of Training Set",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set")))
    fig.write_image("../exercises/mse.over.training.percentage.png")

if __name__ == '__main__':

    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset

    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response

    feature_evaluation(X, y, "../exercises/house price corr")

    # Question 3 - Split samples into training- and testing sets.

    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    #TODO uncomment:
    # question_4()

    x = np.array([1,2,3,4])
    k = 4
    # print(np.vander(x, k)[::-1])
    # print(np.vander(x, k))
    print(np.vander(x, k))
    print(np.flip(np.vander(x, k), 1))
    print(np.vander(x, k, increasing=True))