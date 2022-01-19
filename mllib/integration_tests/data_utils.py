import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from consts import TEST_DATA_PATH


def get_data_dir(file_name):
    return os.path.join(TEST_DATA_PATH, file_name)


def get_mnist_data(should_shuffle=True, should_plot_examples=True):
    print("Reading in and transforming data...")
    df = pd.read_csv(get_data_dir('mnist.csv'))
    data = df.values
    if (should_shuffle == True):
        np.random.shuffle(data)
    X = np.divide(data[:, 1:], 255.0) # data is from 0..255
    Y = data[:, 0]
    picture_shape = (28, 28)

    if (should_plot_examples == True):
        plot_examples(X.reshape((-1, *picture_shape)), Y, cmap='gray', labels=None)
    return X, Y, picture_shape


def get_xor_data(N=200, should_plot_data=True):
    X = np.zeros((N, 2))
    Nq = N // 4
    X[:Nq] = np.random.random((Nq, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    X[Nq:2 * Nq] = np.random.random((Nq, 2)) / 2  # (0-0.5, 0-0.5)
    X[2 * Nq:3 * Nq] = np.random.random((Nq, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    X[3 * Nq:] = np.random.random((Nq, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    Y = np.array([0] * (N // 2) + [1] * (N // 2))

    X, Y = shuffle_pairs(X, Y)

    if (should_plot_data == True):
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5);
        plt.title('Training data plot')
        plt.show()
    return X, Y


def get_donut_data(N=200, should_plot_data=True):
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))

    X, Y = shuffle_pairs(X, Y)

    if (should_plot_data == True):
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5);
        plt.title('Training data plot')
        plt.show()
    return X, Y


def shuffle_pairs(X, Y):
    N = len(X)
    indexes = np.arange(N)
    np.random.shuffle(indexes)
    X = X[indexes]
    Y = Y[indexes]
    return X, Y


def plot_misclasified_examples(x, true_lables, predicted_lables, n=5, print_misclassified=False, labels=None):
  misclassified_idx = np.where(predicted_lables != true_lables)[0]
  misclassified_random_idxes = np.random.choice(misclassified_idx, n*n)
  plt.figure(figsize=(15,15))
  for i in range(n*n):
      idx = misclassified_random_idxes[i]
      plt.subplot(n,n,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x[idx], cmap='gray')
      if labels==None:
        plt.xlabel("True  %s, Pred: %s" % (true_lables[idx], predicted_lables[idx]))
      else:
        plt.xlabel("True  %s, Pred: %s" % (labels[true_lables[idx]], labels[predicted_lables[idx]]))
  plt.show()

  if print_misclassified:
      if labels==None:
        print(pd.DataFrame({'idx':misclassified_random_idxes,
                        'true':true_lables[misclassified_random_idxes],
                        'pred':predicted_lables[misclassified_random_idxes]}))
      else:
        print(pd.DataFrame({'idx':misclassified_random_idxes,
                        #'true':labels[true_lables[misclassified_random_idxes]],
                        'true':true_lables[misclassified_random_idxes],
                        #'pred':labels[predicted_lables[misclassified_random_idxes]]}))
                        'pred':predicted_lables[misclassified_random_idxes]}))


def plot_examples(x, y, cmap='gray', labels=None):
  plt.figure(figsize=(15,15))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(x[i], cmap=cmap)
      if labels==None:
        plt.xlabel(y[i])
      else:
        plt.xlabel(labels[y[i]])
  plt.show()
