import numpy as np

def train_test_split(X, y, test_size=None, train_size=None, random_state=None):
    """
    Split your dataset into training and testing sets easily.

    Args:
    - X (array-like): Your data features (should be in a 2D array format or a 1D array format).
    - y (array-like): Your target labels (should be in a 1D array format).
    - test_size (float, optional): How much data you want to set aside for testing (0 to 1).
    - train_size (float, optional): How much data you want to keep for training (0 to 1).
    - random_state (int, optional): Set a random seed for consistent results.

    Returns:
    - X_train, X_test, y_train, y_test: Your split datasets.

    Quick Notes:
    - If you don't specify test_size or train_size, it defaults to the classic 80% training and 20% testing split.
    - If you set one size, the other will adjust to make sure they add up to 100%.
    - Both sizes should be between 0 and 1 (e.g., 0.2 for 20%). They can't both be 1 as that means no split.
    - If something doesn't add up or if you try to use zero or negative sizes, we'll let you know with an error message.
    - We shuffle your data to mix it up before splitting for randomness.

    Enjoy splitting your data and have fun training your models!
    """
    if random_state is not None:
        np.random.seed(random_state)

    #test_size and train_size are zero
    if (test_size == 0.0 and train_size == 0.0):
        print("Oops! Both test_size and train_size are set to 0. Please provide valid sizes.")
        return None, None, None, None

    #number of dimensions of X
    num_dimensions_X = X.ndim if hasattr(X, 'ndim') else 0

    #X is a one-dimensional array and reshape it if necessary
    if num_dimensions_X == 1:
        X = X.reshape(-1, 1)

    # Default sizes
    if test_size is None and train_size is None:
        test_size = 0.2
        train_size = 0.8
    elif test_size is None:
        test_size = 1 - train_size
    elif train_size is None:
        train_size = 1 - test_size

    try:
        #sizes add up to 1
        if train_size + test_size != 1.0:
            raise ValueError("Oops! The training and testing sizes should add up to 1.")

        if test_size <= 0.0 or train_size <= 0.0:
            raise ValueError("Uh-oh! The training or testing size can't be zero or negative.")

        if test_size == 1.0 and train_size == 1.0:
            raise ValueError("Hey there! Both training and testing can't be 100%. Gotta split it somehow!")

        #shuffling 
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        #splitting indices for train and test
        split_index = int(len(X) * train_size)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        #train and test sets
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

    except ValueError as e:
        print("Oops! Something went wrong:", e)
        return None, None, None, None
    #return output
    return X_train, X_test, y_train, y_test