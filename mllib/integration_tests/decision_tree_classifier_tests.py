import numpy as np
import pandas as pd
from datetime import datetime

from decision_tree_estimator import BinaryTreeClassifier
from integration_tests.utils.data_utils import get_mnist_data, get_xor_data, get_donut_data


def run_model(X, Y, N=2000, max_depth=10, max_bucket_size=10, trace_logs=True):
    X_train, X_test = X[:N // 2], X[N // 2:N]
    Y_train, Y_test = Y[:N // 2], Y[N // 2:N]

    model = BinaryTreeClassifier(max_depth, max_bucket_size, trace_logs=trace_logs)

    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print(f'Fitted within {datetime.now() - t0} time')

    t0 = datetime.now()
    acc = model.score(X_train, Y_train)
    print(f'Predicted within {datetime.now() - t0} time with train accuracy={acc}')

    t0 = datetime.now()
    acc = model.score(X_test, Y_test)
    print(f'Predicted within {datetime.now() - t0} time with test accuracy={acc}')

    return model


if __name__ == '__main__':
    #  for MNIST
    X, Xtest, Y, Ytest, _pic_shape = get_mnist_data()
    idxs = np.logical_or(Y == 0, Y == 1)
    X = X[idxs].copy()
    Y = Y[idxs].copy()

    model = run_model(X, Y, N=10000, max_depth=7, max_bucket_size=None)
    imp = pd.DataFrame(model.get_importance(), columns=['columns_id', 'imporatnce'])
    sorted_imp = imp.sort_values(by='imporatnce', ascending=False)
    print(f'Imporatnce:{sorted_imp}')

    # for xor
    N = 20000
    X, Y = get_xor_data(N)

    model = run_model(X, Y, N=N, max_depth=10, max_bucket_size=None)
    imp = pd.DataFrame(model.get_importance(), columns=['columns_id', 'imporatnce'])
    sorted_imp = imp.sort_values(by='imporatnce', ascending=False)
    print(f'Imporatnce:{sorted_imp}')

    # for donut
    N = 20000
    X, Y = get_donut_data(N)

    model = run_model(X, Y, N=N, max_depth=20, max_bucket_size=None)
    imp = pd.DataFrame(model.get_importance(), columns=['columns_id', 'imporatnce'])
    sorted_imp = imp.sort_values(by='imporatnce', ascending=False)
    print(f'Imporatnce:{sorted_imp}')
