import pandas as pd

def simple_linear_regression(x, y):
    """
    Parameters:
    x (list or array-like): Independent variable data.
    y (list or array-like): Dependent variable data.
    
    Returns:
    result_df (DataFrame): DataFrame containing various calculated values.
    sst (float): Total Sum of Squares.
    sse (float): Sum of Squares of Residuals.
    ssr (float): Sum of Squares of Regression.
    beta0 (float): Coefficient beta0 (slope).
    beta1 (float): Coefficient beta1 (intercept).
    beta2 (float): Coefficient beta2 (associated with 'xy').
    xy_mean (float): Mean of the 'xy' column.
    correlation_xy_y (float): Correlation between 'xy' and 'y'.
    """
    
    #dataframe from x and y
    data = {'x': x, 'y': y}
    df = pd.DataFrame(data)

    #product of 'xy' column
    df['xy'] = df['x'] * df['y']

    #mean of x and y
    y_mean = df['y'].mean()
    x_mean = df['x'].mean()

    #SST (Total sum of square)
    sst = ((df['y'] - y_mean) ** 2).sum()
    
    #differences from the mean
    df['xi - x mean'] = df['x'] - x_mean
    df['yi - y mean'] = df['y'] - y_mean

    #sums of squares
    df['(xi - x mean)(yi - y mean)'] = df['xi - x mean'] * df['yi - y mean']
    df['(xi - x mean)**2'] = df['xi - x mean'] * df['xi - x mean']

    #beta0 (slope) and beta1 (intercept) using formulas
    beta0 = (df['(xi - x mean)(yi - y mean)'].sum()) / (df['(xi - x mean)**2'].sum())
    beta1 = y_mean - beta0 * x_mean

    #prediction
    predicted_values = beta1 + beta0 * df['x']
    df['Prediction'] = predicted_values
    #residual error columns
    df['Residual'] = df['y'] - df['Prediction']
    #SSE (Sum of squares of Residuals)
    sse = (df['Residual'] ** 2).sum()

    #SSR (sum of square of regression) 
    df['(y pred - y mean)**2'] = (df['Prediction'] - y_mean) ** 2
    ssr = df['(y pred - y mean)**2'].sum()

    #dataframe with all the calculated values
    result_data = {
        #independent variable
        'x': df['x'],
        #dependent variable
        'y': df['y'],
        #difference from the mean
        'xi - x mean': df['xi - x mean'],
        'yi - y mean': df['yi - y mean'],
        #sum of squares
        '(xi - x mean)(yi - y mean)': df['(xi - x mean)(yi - y mean)'],
        '(xi - x mean)**2': df['(xi - x mean)**2'],
        #predictions
        'Prediction': df['Prediction'],
        #residual error
        'Residual': df['Residual'],
        #sum of square of regression
        '(y pred - y mean)**2': df['(y pred - y mean)**2'],
        #square of x
        'x square': df['x'] ** 2,
        #square of y
        'y square': df['y'] ** 2,
        #product of x and y
        'xy': df['xy']
    }

    result_df = pd.DataFrame(result_data)
    #mean of the 'xy' column
    xy_mean = df['xy'].mean()
    #the correlation between 'xy' and 'y'
    correlation_xy_y = df['xy'].corr(df['y'])
    #the coefficient beta2 (associated with 'xy')
    beta2 = (df['(xi - x mean)(yi - y mean)'].sum()) / (df['xy'] ** 2).sum()
    #r-squared 
    r_squared = 1 - (sse / sst)
    #mean squared error (MSE)
    mse = sse / (len(x) - 2)  # Degrees of freedom = n - 2 for simple linear regression
    #root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    
    #SST (Sum of Squared Total)
    #Sum of Square Residual Error
    #Total Sum of Square Regression
    #Coefficient beta0 (slope)
    #Coefficient beta1 (intercept)
    #Coefficient beta2 (associated with 'xy')
    return return result_df, sst, sse, ssr, beta0, beta1, beta2, xy_mean, correlation_xy_y, r_squared, mse, rmse
    
#result_df, sst, sse, ssr, beta0, beta1, beta2, xy_mean, correlation_xy_y, r_squared, mse, rmse = simple_linear_regression(x,y)