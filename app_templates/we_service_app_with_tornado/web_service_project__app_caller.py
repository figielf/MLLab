import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from requests import Response


def get_mnist_data(should_shuffle = True):
  mnist_data = pd.read_csv('C:\dev\my_private\MLLab\lazyprogrammer\data\mnist.csv', header='infer').values
  if (should_shuffle == True):
    np.random.shuffle(mnist_data)
  Y_mnist = mnist_data[:, 0]
  X_mnist = np.divide(mnist_data[:, 1:], 255.0)
  return X_mnist, Y_mnist


X, Y = get_mnist_data(should_shuffle=False)
N = len(X)
while True:
  i = np.random.choice(N)
  responce = requests.post('http://localhost:8889/predict', data={'input':X[i]})
  responce_as_json = responce.json()
  print('Prediction service response:', responce_as_json)
  print('Target:', Y[i])

  plt.figure(figsize=(10,10))
  plt.imshow(X[i].reshape(28, 28), cmap='gray')
  plt.show()

  sign = raw_imput('Continue y/n?')
  if (sign in ('n', 'N')):
    break
