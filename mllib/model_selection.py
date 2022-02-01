import numpy as np
from sklearn.model_selection import KFold


def my_cross_val_score(estimator, X, Y, cv, shuffle=False, random_state=None):
    if isinstance(cv, KFold):
        kf = cv
    elif isinstance(cv, int):
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=shuffle)
    else:
        raise Exception(f'cv param can be int or KFold but was {type(cv)} and had value of {cv}')

    kf_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        estimator.fit(X_train, Y_train)
        estimator.score(X_test, Y_test)
        kf_scores.append(estimator.score(X_test, Y_test))
    return np.array(kf_scores)
