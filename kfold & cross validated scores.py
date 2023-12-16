import numpy as np

#function to calculate mean absolute error
def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
    
#function for mean square error
def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

#function for root mean square error
def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

#function for r square
def r_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

#function for kfold validation
def kfold(data, k, shuffle=True, random_state=None):
    #checking the number of folds is at least 2, otherwise raise an error
    if k < 2:
        #print number of fold and it must be at least 2
        raise ValueError("Number of folds must be at least 2")
    #total length of the dataset
    num_samples = len(data)
    #an array of indices from 0 to the number of sample in dataset
    indices = np.arange(num_samples)
    #shuffle the indices if shuffle is True
    if shuffle:
        #random number generator with the given random state
        rng = np.random.default_rng(random_state)
        #shuffle the indices array
        rng.shuffle(indices)
    #an array to store the size of each fold
    fold_sizes = np.full(k, num_samples // k, dtype=int)
    #distributing the remainder among the folds if the data size isn't divisible evenly
    fold_sizes[:num_samples % k] += 1
    #a variable to tracking the current index
    current = 0

    #arrays to store training and testing indices for each fold
    train_indices = np.zeros((k, num_samples - num_samples // k), dtype=int)
    test_indices = np.zeros((k, num_samples // k), dtype=int)

    #iterating over each fold
    for i, fold_size in enumerate(fold_sizes):
        #start and stop indices for the current fold
        start, stop = current, current + fold_size
        #test indices for the current fold
        test_idx = indices[start:stop]
        #training indices for the current fold
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        #training and testing indices in their respective arrays
        train_indices[i, :len(train_idx)] = train_idx
        test_indices[i, :len(test_idx)] = test_idx
        #update the current index
        current = stop
    #training and testing indices arrays
    return train_indices, test_indices

#function for cross validation
def cross_validated_score(model, X, y, cv=5, scoring=None):
    #generate train and test indices
    train_indices, test_indices = kfold(X, k=cv)
    #store each fold
    scores = []
    #for loop to iterate cv
    for i in range(cv):
        #trianing and testing indices for the current fold
        train_index, test_index = train_indices[i], test_indices[i]
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        #fitting or training model 
        model.fit(X_train_fold, y_train_fold)
        #predicting y value
        predictions = model.predict(X_test_fold)
        #selecting score types
        score = scoring(y_test_fold, predictions)
        scores.append(score)
        #printing all scores
        print(f"Score for fold {i + 1}: {score}")
    #mean value of all score
    mean_score = np.mean(scores)
    print()
    #prining mean of all score
    print(f"Mean score: {mean_score}")


# kf = kfold(X_train, 10, shuffle=False, random_state=0)
# cross_validated_score(line_model, X_train, y_train, cv=10, scoring=r_squared)
