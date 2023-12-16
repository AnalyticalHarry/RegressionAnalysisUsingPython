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

#function for k-fold validation
def kfold(data, k, shuffle=True, random_state=None):
    #number of folds is at least 2
    if k < 2:
        #print number of fold and it must be at least 2
        raise ValueError("Number of folds must be at least 2")
    
    #total number of length in data
    num_samples = len(data)
    
    #an array of indices from 0 to total number of data length
    indices = np.arange(num_samples)
    
    #if shuffle is true
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    
    #size of each fold and distribute any remainder
    fold_sizes = np.full(k, num_samples // k, dtype=int)
    fold_sizes[:num_samples % k] += 1
    
    #current index, set to zero
    current = 0

    #training and testing indices for each fold
    train_indices = np.zeros((k, num_samples - num_samples // k), dtype=int)
    test_indices = np.zeros((k, num_samples // k), dtype=int)

    #iterate over each fold
    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        train_indices[i, :len(train_idx)] = train_idx
        test_indices[i, :len(test_idx)] = test_idx
        current = stop

    return train_indices, test_indices

#function for cross-validation 
def cross_validated_score(model, X, y, cv=5, scoring=None):
    #train and test indices using k-fold
    train_indices, test_indices = kfold(X, k=cv)
    #code to store scores
    scores = []

    #loop thorugh each fold
    for i in range(cv):
        train_index, test_index = train_indices[i], test_indices[i]
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        #fit and train model
        model.fit(X_train_fold, y_train_fold)
        #predictions on the testing data for the current fold
        predictions = model.predict(X_test_fold)
        #evluating score from various metrics
        score = scoring(y_test_fold, predictions)
        scores.append(score)
    
    #total mean of all fold
    mean_score = np.mean(scores)
    print(f'Mean Value for scores: {mean_score}')
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(range(1, len(scores) + 1), scores, color='blue', alpha=0.7)
    ax.set_title('Scores for Each Fold')
    ax.set_ylabel('Fold')
    ax.set_yticks(range(1, len(scores) + 1))
    for i, score in enumerate(scores):
        ax.text(score, i + 1, f'{score:.2f}', ha='left', va='center')
    ax.set_xlabel('Score')
    ax.grid(True, ls='--', alpha=0.3, color='black')
    plt.tight_layout()
    plt.show()
    
    return scores


# kf = kfold(X_train, 10, shuffle=False, random_state=0)
# cross_validated_score(line_model, X_train, y_train, cv=10, scoring=r_squared)
