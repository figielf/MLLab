import numpy as np
import pandas as pd
from datetime import datetime
from data_utils import get_mnist_data, get_xor_data, get_donut_data
from decision_tree_classifier import BinaryTreeClassifier


def run_model(X, Y, N=2000, max_depth=10, max_bucket_size=10, trace_logs=True):
    X_train, X_test = X[:N // 2], X[N // 2:N]
    Y_train, Y_test = Y[:N // 2], Y[N // 2:N]

    model = BinaryTreeClassifier(max_depth, max_bucket_size, trace_logs=trace_logs)

    # print('Y_train:', Y_train)
    t0 = datetime.now()
    model.fit(X_train, Y_train)
    print(f'Fitted within {datetime.now() - t0} time')

    t0 = datetime.now()
    acc = model.score(X_train, Y_train)
    print(f'Predicted within {datetime.now() - t0} time with train accuracy={acc}')

    t0 = datetime.now()
    acc = model.score(X_test, Y_test)
    print(f'Predicted within {datetime.now() - t0} time with test accuracy={acc}')

    imp = pd.DataFrame(model.get_importance(), columns=['columns_id', 'imporatnce'])
    sorted_imp = imp.sort_values(by='imporatnce', ascending=False)
    print(f'Imporatnce:{sorted_imp}')


def _sanity_test():
  x=np.array([1,4,-999,1,2,8,19,2,7,16,3,1,34,12,19,2,1111,55,3,-9898,24,9,-100,4]).reshape((-1,3))
  y=np.array([0,1,1,1,0,1,0,1])
  #x=np.array([7,7,7,7,7,7,7,7,7,7]).reshape((-1,1))
  #y=np.array([0,1,1,1,0,0,1,1,1,0])
  #x_pred=np.array([1,1,1,1,1]).reshape((-1,1))
  #y_pred=np.array([0,1,1,1,])

  model = BinaryTreeClassifier(5, 2)
  model.fit(x, y)

  print('train accuracy:', (model.predict(x) == y).mean())

  imp = model.get_importance()
  print(f'test result imporatnce:{imp}')


if __name__ == '__main__':
    print('Sanity test:')
    _sanity_test()

    print('MNIST data test:')
    X, Y, _ = get_mnist_data()
    model = run_model(X, Y, N=10000, max_depth=7, max_bucket_size=None, trace_logs=True)

    print('MNIST data test:')
    N=10000
    X, Y = get_donut_data(N, )
    model = run_model(X, Y, N=N, max_depth=7, max_bucket_size=None, trace_logs=False)

    print('MNIST data test:')
    N=10000
    X, Y = get_xor_data()
    model = run_model(X, Y, N=N, max_depth=7, max_bucket_size=None, trace_logs=False)
