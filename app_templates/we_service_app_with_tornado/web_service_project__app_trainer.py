import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_mnist_data(should_shuffle = True):
  mnist_data = pd.read_csv('C:\dev\my_private\MLLab\lazyprogrammer\data\mnist.csv', header='infer').values
  if (should_shuffle == True):
    np.random.shuffle(mnist_data)
  Y_mnist = mnist_data[:, 0]
  X_mnist = np.divide(mnist_data[:, 1:], 255.0)
  return X_mnist, Y_mnist


X, Y = get_mnist_data(should_shuffle=False)
N = len(X)//4
Xtrain, Ytrain = X[:N], Y[:N]
Xtest, Ytest = X[N:], Y[N:]

model = RandomForestClassifier()
model.fit(Xtrain, Ytrain)

print('Test accuracy:', model.score(Xtest, Ytest))

with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print('Trained model saved in model.pkl file')
