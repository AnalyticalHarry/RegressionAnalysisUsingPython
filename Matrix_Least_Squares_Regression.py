import numpy as np
from IPython.display import display, HTML

def tab(data):
    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        columns = list(data[0].keys())
        rows = [list(row.values()) for row in data]
    elif isinstance(data, list) and all(isinstance(d, list) for d in data):
        columns = data[0]
        rows = data[1:]
    else:
        raise ValueError("Input data should be a list of dictionaries or a list of lists.")

    table_html = "<table>"
    table_html += "<tr>"
    for col in columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr>"

    for row in rows:
        table_html += "<tr>"
        for col in row:
            table_html += f"<td>{col}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    display(HTML(table_html))

class Matrix_Least_Squares_Regression:
      """
          Class for simple linear regression.
    
          Attributes:
            X (numpy.ndarray): The independent variable.
            y (numpy.ndarray): The dependent variable.
            intercept (float): Intercept of the regression line (β0).
            slope (numpy.ndarray): Slope(s) of the regression line (β1).
            predicted (numpy.ndarray): Predicted values.
            residual (numpy.ndarray): Residuals.
            y_mean (float): Mean of the dependent variable.
            SST (float): Total sum of squares.
            SSR (float): Regression sum of squares.
            r_squared (float): Coefficient of Determination (R-squared).
            mse (float): Mean Squared Error.
            rmse (float): Root Mean Squared Error.
       """
    def fit(self, X, y):
        """
        Fit a simple linear regression model.

        Parameters:
            X (numpy.ndarray): The independent variable.
            y (numpy.ndarray): The dependent variable.
        """
        X = np.array(X)
        y = np.array(y)
        # Check if X is 2D and y is 1D
        if len(X.shape) != 2:
            raise ValueError("X should be a 2D array")
        if len(y.shape) != 1:
            raise ValueError("y should be a 1D array")

        # Add intercept term to X
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ y
        self.intercept, self.slope = self.beta[0], self.beta[1:]
        self.predicted = self.intercept + np.dot(X, self.slope)
        self.residual = self.predicted - y
        self.y_mean = np.mean(y)
        self.SST = np.sum((y - self.y_mean) ** 2)
        self.SSR = np.sum((self.predicted - self.y_mean) ** 2)
        self.r_squared = self.SSR / self.SST
        self.mse = np.mean((y - self.predicted) ** 2)
        self.rmse = np.sqrt(self.mse)

    def predict(self, X):
        X = np.array(X)
        return self.intercept + np.dot(X, self.slope)

    def model_evaluation(self):
        result_data = [
            {"Parameter": "Intercept", "Value": round(self.intercept, 4)}
        ]
        for i, slope in enumerate(self.slope, 1):
            result_data.append({"Parameter": f"Slope {i}", "Value": round(slope, 4)})
        result_data.extend([
            {"Parameter": "R-squared", "Value": round(self.r_squared, 4)},
            {"Parameter": "Mean Squared Error", "Value": round(self.mse, 4)},
            {"Parameter": "Root Mean Squared Error", "Value": round(self.rmse, 4)}
        ])
        tab(result_data)