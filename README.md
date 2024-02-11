# Regression Model for Correlation and Trend Analysis 
[![YouTube](https://img.shields.io/badge/AnalyticalHarry-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@AnalyticalHarry)
[![Website](https://img.shields.io/badge/topmate.io-AnalyticalHarry-blue?style=for-the-badge&logo=web)](https://topmate.io/analyticalharry)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-AnalyticalHarry-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/analyticalharry/)
[![Twitter](https://img.shields.io/badge/Twitter-AnalyticalHarry-blue?style=for-the-badge&logo=twitter)](https://twitter.com/AnalyticalHarry)

Regression analysis is a cornerstone of statistical methods used for predicting a continuous outcome variable based on one or more predictor variables. It offers insights into the relationships between variables, enabling both prediction and inference. This document covers key aspects of regression analysis, including various types of regression, methods for estimation, residual analysis, and more.

## Linear Regression

Linear regression models the linear relationship between a dependent variable and independent variables X. 

### Ordinary Least Squares (OLS)

OLS is the most prevalent method for estimating the parameters of a linear regression model, aiming to minimize the sum of squared differences between observed and predicted values.

### Weighted Least Squares (WLS) and Generalized Least Squares (GLS)

- **WLS** addresses heteroscedasticity by assigning weights based on the inverse variance of errors.
- **GLS** extends WLS to accommodate correlated error terms.

### Lasso and Ridge Regression

These methods introduce penalties to the regression model to prevent overfitting:

- **Lasso (L1 Regularization)**: Adds the absolute value of coefficients as a penalty to the loss function.
- **Ridge (L2 Regularization)**: Incorporates the squared magnitude of coefficients into the penalty.

## Residual Analysis

Examining residuals is important for validating regression model assumptions, assessing for patterns that suggest issues like non-linearity or heteroscedasticity.

### Heteroscedasticity vs. Homoscedasticity

- **Heteroscedasticity**: Variance of error terms is not constant.
- **Homoscedasticity**: Desired condition where variance is constant.

### Tests for Heteroscedasticity

- **Breusch-Pagan Test**: Assesses heteroscedasticity by regressing squared residuals on independent variables.
- **White Test**: A general test for heteroscedasticity without assuming a specific form.

## Skewness and Kurtosis

- **Skewness**: Measures the asymmetry of the residuals distribution.
- **Kurtosis**: Indicates the "tailedness" of the residuals distribution.

## Polynomial Features

Adding polynomial features can help model non-linear relationships by including powers or interactions of original variables.

## Gradient Descent

An optimization algorithm used to minimize the cost function, adjusting parameters iteratively to find the cost function's minimum value.
