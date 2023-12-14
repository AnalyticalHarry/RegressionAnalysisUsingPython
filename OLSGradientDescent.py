class OLSGradientDescentRegression:
    def __init__(self, X, y, learning_rate=0.01, 
                 num_iterations=1000, 
                 split=0.8,
                 verbose=False):
        self.verbose = verbose
        if split < 0.0 or split > 1.0:
            raise ValueError("split should be between 0 and 1")
        
        self.X = np.array(X)
        self.y = np.array(y)

        if len(self.X.shape) != 2:
            raise ValueError("X should be a 2D array")
        if len(self.y.shape) != 1:
            raise ValueError("y should be a 1D array")

        if split == 0:
            # Use all data for training if split is 0
            X_train, y_train = self.X, self.y
            X_test, y_test = self.X, self.y
        else:
            # Split the data into training and testing sets
            split_idx = int(split * len(self.X))
            X_train, X_test = self.X[:split_idx], self.X[split_idx:]
            y_train, y_test = self.y[:split_idx], self.y[split_idx:]

        self.X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
        self.X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

        self.beta = np.zeros(self.X_train.shape[1])
        self.m = len(y_train)
        self.y_train = y_train
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        cost_history = []
        r_squared_history = []
        for i in range(self.num_iterations):
            # Train the model using the training data
            predictions_train = np.dot(self.X_train, self.beta)
            error_train = predictions_train - y_train
            gradient = (1 / self.m) * np.dot(self.X_train.T, error_train)
            self.beta -= self.learning_rate * gradient
            cost = np.mean(error_train ** 2)
            cost_history.append(cost)

            # R-squared calculation
            SSR = np.sum(error_train ** 2)
            SST = np.sum((y_train - np.mean(y_train)) ** 2)
            r_squared = 1 - (SSR / SST)
            r_squared_history.append(r_squared)

            if self.verbose and (i % 100 == 0 or i == (self.num_iterations - 1)):
                print(f"Iteration {i}/{self.num_iterations}: Cost = {cost:.4f}, R-squared = {r_squared:.4f}")

        self.intercept, self.slope = self.beta[0], self.beta[1:]
        self.cost_history = cost_history
        self.r_squared_history = r_squared_history

    def predict(self, X):
        # Ensure X is a numpy array
        X = np.array(X)
    
        # If X is a 1D array, reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
    
        # Add a column of ones for the intercept
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        return np.dot(X_with_intercept, self.beta)

    def metrics(self):
        predictions_test = np.dot(self.X_test, self.beta)
        residuals_test = self.y_test - predictions_test
        y_mean_test = np.mean(self.y_test)
        SST_test = np.sum((self.y_test - y_mean_test) ** 2)
        SSR_test = np.sum((predictions_test - y_mean_test) ** 2)
        r_squared_test = SSR_test / SST_test
        mse_test = np.mean(residuals_test ** 2)
        rmse_test = np.sqrt(mse_test)
        mae_test = np.mean(np.abs(residuals_test))

        print(f"Intercept: {self.intercept:.4f}")
        print("Slope:", end=" ")
        for slope_value in self.slope:
            print(f"{slope_value:.4f}", end=" ")
        print()
        print(f"R Square: {r_squared_test:.4f}")
        print(f"Mean Squared Error (MSE): {mse_test:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f}")
        print(f"Mean Absolute Error (MAE): {mae_test:.4f}")

    def cost_function(self):
        plt.plot(range(1, self.num_iterations + 1), self.cost_history)
        plt.xlabel("Iterations")
        plt.ylabel("Cost (MSE)")
        plt.ticklabel_format(style='plain', axis='both')
        plt.title("Gradient Descent")
        plt.grid(True, linestyle='--', alpha=0.2, color='black')
    
        #coordinates of the last point
        last_iteration = self.num_iterations
        last_cost = self.cost_history[-1]
    
        #labelling the last scatter point
        plt.scatter(last_iteration, last_cost, color='red') 
        plt.annotate(f'\nIteration: {last_iteration}\nCost: {last_cost:.4f}', 
                     (last_iteration, last_cost), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
        plt.show()

        
    def r_squared(self):
        plt.plot(range(1, self.num_iterations + 1), self.r_squared_history)
        plt.xlabel("Iterations")
        plt.ylabel("R-squared")
        plt.title("R-squared Over Iterations")
        plt.grid(True, linestyle='--', alpha=0.2, color='black')
    
        # Coordinates of the last point
        last_iteration = self.num_iterations
        last_r_squared = self.r_squared_history[-1]
    
        # Label the last scatter point
        plt.scatter(last_iteration, last_r_squared, color='red')  # Mark the last point with a different color
        plt.annotate(f'\nIteration: {last_iteration}\nR-squared: {last_r_squared:.4f}', 
                     (last_iteration, last_r_squared), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
        plt.show()

        
    def regression_plot(self):
        plt.scatter(self.X_train[:, 1], self.y_train, color='black', alpha=0.7, label='Train')
        plt.scatter(self.X_test[:, 1], self.y_test, color='blue', alpha=0.3, label='Test')
        plt.xlabel("Features (X)")
        plt.ylabel("Target (y)")
        plt.title("Predictive Modeling")
        # Use only the feature columns for prediction, excluding the column of ones
        predictions_test = self.predict(self.X_test[:, 1:])
        plt.plot(self.X_test[:, 1], predictions_test, label='Best Fit Line', color='r')
        plt.grid(True, linestyle='--', alpha=0.2, color='black')
        plt.legend()
        plt.show()
        
    def residual_plot(self):
        predictions_train = np.dot(self.X_train, self.beta)
        residuals_train = self.y_train - predictions_train
        predictions_test = np.dot(self.X_test, self.beta)
        residuals_test = self.y_test - predictions_test
        plt.scatter(predictions_train, residuals_train, color='black', alpha=0.5, label='Train')
        plt.scatter(predictions_test, residuals_test, color='blue', alpha=0.5, label='Test')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.2, color='black')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.legend()
        plt.show()

