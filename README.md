# About
First attempt in machine learning on Time series forecasting.

This is part of a course project to determine the performance between traditional forecasting methods and machine learning algorithms in time-series forecasting.

## Objective of the project

A comparison between machine learning algorithms and traditional forecasting methods to predict the store sales on data from a large grocery retailer. 
<br>

We will use data from the kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

The goal is to use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer.

Specifically, we have to build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores.

In this project, we will do a baseline comparison between the two methods, hence, we will only use a subset of the dataset, a top selling product family from a store with a lot of sales. 


## Forecasting Methods

> Traditional Forecasting Methods
* Simple Exponential Smoothing
* Double Exponential Smoothing
* Triple Exponential Smoothing
* SARIMA
* SARIMAX

> Machine Learning Algorithms
* Linear Regression
* Light Gradient Boosting 
* Random Forest
* XGBoost

## Metrics Used
* RMSE
* MAE


A baseline notebook was forked from [Forecasting Wars - Classical Forecasting Methods vs Machine Learning](https://github.com/Deffro/Data-Science-Portfolio/blob/master/Notebooks/Forecasting%20Wars%20-%20Classical%20Forecasting%20Methods%20vs%20Machine%20Learning/Forecasting%20Wars%20-%20Classical%20Forecasting%20Methods%20vs%20Machine%20Learning.ipynb) by [Deffro](https://github.com/Deffro) and some changes were made to fit our dataset with additional ML algorithm. 


References:<br>

* https://www.analyticsvidhya.com/blog/2022/06/time-series-forecasting-using-python/
* https://www.kaggle.com/code/ekrembayar/store-sales-ts-forecasting-a-comprehensive-guide
* https://towardsdatascience.com/forecasting-wars-classical-forecasting-methods-vs-machine-learning-4fd5d2ceb716
