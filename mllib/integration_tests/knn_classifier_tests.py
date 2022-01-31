from datetime import datetime
from integration_tests.utils.data_utils import get_mnist_data, get_xor_data, get_donut_data
from knn_classifier_estimator import KnnClassifier


def run_model(k, X, Y, N=2000):
    X_train, X_test = X[:N // 2], X[N // 2:N]
    Y_train, Y_test = Y[:N // 2], Y[N // 2:N]

    model = KnnClassifier(k)

    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print(f'Fitted within {datetime.now() - t0} time')

    t0 = datetime.now()
    acc = model.score(X_train, Y_train)
    print(f'Predicted within {datetime.now() - t0} time with train accuracy={acc}')

    t0 = datetime.now()
    acc = model.score(X_test, Y_test)
    print(f'Predicted within {datetime.now() - t0} time with test accuracy={acc}')


if __name__ == '__main__':
    print('MNIST data test:')
    X, Y, _ = get_mnist_data()
    N = 2000
    for k in range(1, 6):
        print(f'{k}NN classifier')
        run_model(k, X.copy(), Y.copy(), N)

    print('MNIST data test:')
    N=2000
    X, Y = get_donut_data(N)
    for k in range(1, 6):
        print(f'{k}NN classifier')
        run_model(k, X.copy(), Y.copy(), N)

    print('MNIST data test:')
    N=2000
    X, Y = get_xor_data(N)
    for k in range(1, 6):
        print(f'{k}NN classifier')
        run_model(k, X.copy(), Y.copy(), N)
