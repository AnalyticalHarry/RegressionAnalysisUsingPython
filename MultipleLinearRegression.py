import pandas as pd
import math

def multiple_linear_regression(x, y):
    """
    Parameters:
    x (array-like): Independent variables (features) as a 2D array.
    y (array-like): Dependent variable (target) as a 1D array.

    Returns:
    result_df (DataFrame): DataFrame containing various calculated values.
    sst (float): Total Sum of Squares.
    sse (float): Sum of Squares of Residuals.
    ssr (float): Sum of Squares of Regression.
    betas (list): Coefficients (beta values).
    r_squared (float): Coefficient of Determination (R-squared).
    mse (float): Mean Squared Error.
    rmse (float): Root Mean Squared Error.
    """
    #dataframe from the independent variables x and the dependent variable y
    data = {f'x{i}': x[:, i] for i in range(x.shape[1])}
    data['y'] = y
    df = pd.DataFrame(data)

    #mean of y and each independent variable
    y_mean = df['y'].mean()
    x_means = [df[f'x{i}'].mean() for i in range(x.shape[1])]

    #total sum of squares (SST)
    sst = ((df['y'] - y_mean) ** 2).sum()

    #coefficients (beta values)
    betas = [0] * (x.shape[1] + 1)  # +1 for the intercept
    
    #differences from the means
    for i in range(x.shape[1]):
        df[f'x{i} - x{i}_mean'] = df[f'x{i}'] - x_means[i]

    #sums of squares
    for i in range(x.shape[1]):
        df[f'(x{i} - x{i}_mean) * (y - y_mean)'] = df[f'x{i} - x{i}_mean'] * (df['y'] - y_mean)
        df[f'(x{i} - x{i}_mean)**2'] = df[f'x{i} - x{i}_mean'] ** 2

    #beta values using formulas
    for i in range(x.shape[1]):
        betas[i + 1] = (df[f'(x{i} - x{i}_mean) * (y - y_mean)'].sum()) / (df[f'(x{i} - x{i}_mean)**2'].sum())

    betas[0] = y_mean - sum(betas[i + 1] * x_means[i] for i in range(x.shape[1]))

    #predictions
    df['Prediction'] = betas[0] + sum(betas[i + 1] * df[f'x{i}'] for i in range(x.shape[1]))

    #residuals
    df['Residual'] = df['y'] - df['Prediction']

    #sum of squares of residuals (SSE)
    sse = (df['Residual'] ** 2).sum()

    #Sum of Squares of Regression (SSR)
    df['(Prediction - y_mean)**2'] = (df['Prediction'] - y_mean) ** 2
    ssr = df['(Prediction - y_mean)**2'].sum()

    #r-squared (R^2)
    r_squared = 1 - (sse / sst)

    #mean squared error (MSE)
    #degrees of freedom = n - k - 1 for multiple linear regression
    mse = sse / (len(x) - x.shape[1] - 1)  

    #root mean squared error (RMSE)
    rmse = math.sqrt(mse)

    #calculated values
    result_data = {f'x{i}': df[f'x{i}'] for i in range(x.shape[1])}
    result_data['y'] = df['y']
    result_data['Prediction'] = df['Prediction']
    result_data['Residual'] = df['Residual']
    #all results
    result_df = pd.DataFrame(result_data)
    #SST (Sum of Squared Total)
    #Sum of Square Residual Error
    #Total Sum of Square Regression
    #Coefficient beta
    #r squared
    #mean square error
    #root mean square error
    return result_df, sst, sse, ssr, betas, r_squared, mse, rmse
    
# result_df, sst, sse, ssr, betas, r_squared, mse, rmse = multiple_linear_regression(x, y)
# print("SST:", sst)
# print("SSE:", sse)
# print("SSR:", ssr)
# print("Beta Values:", betas)
# print("R-squared (R^2):", r_squared)
# print("Mean Squared Error (MSE):", mse)
# print("Root Mean Squared Error (RMSE):", rmse)