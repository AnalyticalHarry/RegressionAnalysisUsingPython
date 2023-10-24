import numpy as np
import pandas as pd

def simple_linear_regression_matrix(x, y):
    """
    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.

    Returns:
    result_df (DataFrame): DataFrame containing various calculated values.
    intercept (float): Intercept of the regression line (β0).
    slope (float): Slope of the regression line (β1).
    r_squared (float): Coefficient of Determination (R-squared).
    mse (float): Mean Squared Error.
    rmse (float): Root Mean Squared Error.
    """

    #designing matrix X
    X = np.column_stack((np.ones(len(x)), x))
    #coefficients β
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    #intercept and slope
    intercept, slope = beta[0], beta[1]
    #predictions
    predicted = intercept + slope * x
    #residuals
    residual = predicted - y
    #target or dependent mean value
    y_mean = np.mean(y)
    #SST (Sum of Squared Total)
    SST = np.sum((y - y_mean) ** 2)
    #SSR (sum of square of regression) 
    SSR = np.sum((predicted - y_mean) ** 2)
    #r-squared (R²)
    r_squared = SSR / SST
    #mean squared error (MSE)
    mse = np.mean((y - predicted) ** 2)
    #root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    #results
    result_data = {
        'x': x,
        'y': y,
        'Prediction': predicted,
        'Residual': residual
    }
    result_df = pd.DataFrame(result_data)
    
    #slope
    #intercept
    #r squared
    #mean square error
    #root mean square error
    #SST (Total sum of square)
    #SSR (sum of square of regression) 
    return result_df, intercept, slope, r_squared, mse, rmse, SST, SSR


    
# result_df, intercept, slope, r_squared, mse, rmse, SST, SSR = simple_linear_regression_matrix(x, y)
# print("Intercept (β0):", intercept)
# print("Slope (β1):", slope)
# print("R-squared (R²):", r_squared)
# print("Mean Squared Error (MSE):", mse)
# print("Root Mean Squared Error (RMSE):", rmse)