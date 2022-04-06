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

    pass # debugging

    # Question 5 - Evaluating fitted model on different countries






    # # return values
    # processed_sample_matrix = data
    # response = processed_sample_matrix.Temp
    # processed_sample_matrix.drop("Temp", axis=1, inplace=True)
    #
    # # TODO should add intercept ?
    #
    # return processed_sample_matrix, response
