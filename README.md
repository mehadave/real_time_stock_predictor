# real_time_stock_predictor
Hosted Streamlit App URL: https://real-time-stock-predictor.streamlit.app/

## 1. Project Overview
   
This report outlines the theoretical foundations and technical implementation details behind the Real-Time Stock
Predictor application. The purpose of this tool is to use historical stock price data to predict the next day's closing price
for a selected stock using a basic linear regression model. This project demonstrates the application of machine
learning, data analysis, and software development using Python and Streamlit.

## 2. Machine Learning Concepts
Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more
independent variables. In this project, we use a simple linear regression model to predict the next day's closing stock
price based on the current day's price.
The model follows the equation:
 y = mx + b
Where:
- y is the predicted value (next closing price)
- m is the learned coefficient (slope)
- x is the input feature (current closing price)
- b is the intercept (bias term)
The model is trained using the least squares method, which minimizes the sum of squared differences between
predicted and actual values. The performance of the model can be further evaluated using metrics like R² score or Mean
Squared Error (MSE).

## 3. Python Tech Stack
The following Python libraries are used in the implementation:
1. yfinance: For downloading historical stock data directly from Yahoo Finance.
2. numpy: For numerical computation and data manipulation.
3. scikit-learn: For building and training the Linear Regression model.
4. streamlit: For creating an interactive and user-friendly web interface to run the prediction model.
The application flow is simple:
- Download stock data using `yfinance`
- Prepare features and targets by shifting the 'Close' column
- Train the LinearRegression model using scikit-learn
- Predict the next closing price based on the most recent price
The Streamlit app wraps this process in a clean, web-based user interface.

## 4. Mathematical Foundation
The core mathematical concept is linear regression, which assumes a linear relationship between input and output. This
is based on the Ordinary Least Squares (OLS) method, where the model calculates the best-fit line by minimizing the
residual sum of squares between actual and predicted outputs.
This involves solving for the slope (m) and intercept (b) using the following formulas:
 m = covariance(x, y) / variance(x)
 b = mean(y) - m * mean(x)
These values define the regression line used for prediction.
While this model is simple and interpretable, its limitations include an assumption of linearity and sensitivity to outliers
and market volatility.
