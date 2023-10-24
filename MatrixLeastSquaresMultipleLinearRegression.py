import numpy as np
import pandas as pd

def multiple_linear_regression_matrix(X, y):
    """
    parameters:
    x (array-like): independent variables (design matrix).
    y (array-like): dependent variable.
    
    returns:
    result_df (dataframe): dataframe containing various calculated values.
    betas (array): coefficients (beta values) for all independent variables.
    r_squared (float): coefficient of determination (r-squared).
    mse (float): mean squared error.
    rmse (float): root mean squared error.
    """

    #coefficients β
    betas = np.linalg.inv(X.T @ X) @ X.T @ y
    #predictions
    predicted = X @ betas
    #residuals
    residual = predicted - y
    #target or dependent mean value
    y_mean = np.mean(y)
    #sst (sum of squared total)
    sst = np.sum((y - y_mean) ** 2)
    #ssr (sum of squares of regression)
    ssr = np.sum((predicted - y_mean) ** 2)
    #r-squared (r²)
    r_squared = ssr / sst
    #mean squared error (mse)
    mse = np.mean((y - predicted) ** 2)
    #root mean squared error (rmse)
    rmse = np.sqrt(mse)

    #results
    result_data = {
        'y': y,
        'prediction': predicted,
        'residual': residual
    }
    for i in range(X.shape[1]):
        result_data[f'x{i}'] = X[:, i]
    result_df = pd.DataFrame(result_data)
    
    #slope
    #intercept
    #r squared
    #mean square error
    #root mean square error
    #SST (Total sum of square)
    #SSR (sum of square of regression) 
    return result_df, betas, r_squared, mse, rmse, sst, ssr

# result_df, betas, r_squared, mse, rmse = multiple_linear_regression_matrix(X, y)
# print("Coefficients (Beta Values):", betas)
# print("R-squared (R²):", r_squared)
# print("Mean Squared Error (MSE):", mse)
# print("Root Mean Squared Error (RMSE):", rmse)
