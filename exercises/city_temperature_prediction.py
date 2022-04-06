import math

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
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
    data = data[data["Temp"] > -30]
    data = data[data["Temp"] < 60]

    # new features
    data['DayOfYear'] = pd.to_datetime(data['Date']).dt.dayofyear

    return data


if __name__ == '__main__':

    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset

    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country

    # temperatures in Israel as a function of Day Of Year
    israel_df = df[df['Country'] == 'Israel']
    graphs = []
    #
    for year in israel_df['Year'].unique():
        df_per_year = israel_df[israel_df['Year'] == year]
        y_temp = df_per_year['Temp']
        graphs.append(go.Scatter(x=list(range(1, 366)),
                                 y=y_temp,
                                 mode='markers', name=f'{year}'))
    #
    fig = go.Figure(data=graphs)
    fig.update_layout(
        title="temperatures in Israel as a function of Day Of Year",
        xaxis_title="day of year",
        yaxis_title="temperatures",
        height=600, width=1200)
    fig.write_image("../exercises/temp.per.day.png")

    # temperatures std in Israel as a function of months
    fig = go.Figure(data=[go.Bar(x=list(range(1, 13)),
                                 y=israel_df.groupby(['Month']).std()["Temp"])])
    #
    fig.update_layout(
        title="temperatures std in Israel as a function of months",
        xaxis_title="month",
        yaxis_title="temperatures std")
    fig.write_image("../exercises/temp.std.per.month.png")

    # Question 3 - Exploring differences between countries

    temp_statistic_df = df.groupby(['Country', 'Month']).Temp.agg([np.mean, np.std])
    month_indices = temp_statistic_df.index.get_level_values('Month')
    country_indices = temp_statistic_df.index.get_level_values('Country')
    fig = px.line(temp_statistic_df,
                  x=month_indices,
                  y=temp_statistic_df['mean'],
                  error_y=temp_statistic_df['std'],
                  color=country_indices)
    fig.update_layout(
        title="mean temp +- std of all countries as a func of month",
        xaxis_title="month",
        yaxis_title="mean temp +- std")
    fig.write_image("../exercises/mean.temp.for.each.coumtry.png")

    # Question 4 - Fitting model for different values of `k`

    # Randomly split the dataset into a training set (75%) and test set (25%)
    sample_matrix = israel_df.drop("Temp", axis=1, inplace=False)
    response = israel_df.Temp
    train_X, train_y, test_X, test_y = split_train_test(sample_matrix, response)
    # For every value k ∈ [1,10], fit a polynomial model of degree k using the training set
    loss_list = []
    min_loss, min_k = math.inf, 0
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(train_X["DayOfYear"], train_y)
        # Record the loss of the model over the test set, rounded to 2 decimal places
        loss = round(poly_model._loss(test_X["DayOfYear"], test_y), 2)
        loss_list.append(loss)
        if loss < min_loss: min_loss, min_k = loss, k # ensure the simplest model (min k)
        print(f"loss of {k}-polynomial model over the test set: {loss}")
    print(f"the optimal degree is: {min_k}, resulting in a loss of {min_loss}")
    #plot
    fig = go.Figure(data=[go.Bar(x=list(range(1, 11)),
                                 y=loss_list)])
    fig.update_layout(
        title="loss of polynomial model as a func of polynomial degree k",
        xaxis_title="k",
        yaxis_title="loss of polynomial model")
    fig.write_image("../exercises/loss.over.polynomial.degree.png")

    # Question 5 - Evaluating fitted model on different countries

    train_X, train_y = israel_df.drop('Temp', axis=1), israel_df.Temp
    poly_model = PolynomialFitting(min_k)
    poly_model.fit(train_X["DayOfYear"], train_y)
    loss_list = []

    for country in df['Country'].unique():
        if country != "Israel":
            country_df = df[df['Country'] == country]
            loss = poly_model._loss(country_df["DayOfYear"], country_df.Temp)
            loss_list.append(loss)

    fig = go.Figure(data=[go.Bar(x=df[df['Country'] != 'Israel']['Country'].unique(),
                                 y=loss_list)])
    fig.update_layout(
        title=f"polynomial fitting errors of countries other than Israel"
              f"with the optimal degree {min_k} chosen in Israel subset fitting",
        xaxis_title="country",
        yaxis_title="loss of Polynomial fitting of degree k in israel")
    fig.write_image("../exercises/countries.loss.over.min_k.png")

