import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from ensamble.bagging_estimator import BaggingRegressor

if __name__ == '__main__':
    T = 100
    x_axis = np.linspace(0, 2 * np.pi, T)
    y_axis = np.sin(x_axis)

    N = 30
    idx = np.random.choice(T, size=N, replace=False)
    Xtrain = x_axis[idx].reshape(N, 1)
    Ytrain = y_axis[idx]

    # Single (not bagged) model for comparison
    model = DecisionTreeRegressor()
    model.fit(Xtrain, Ytrain)
    simple_prediction = model.predict(x_axis.reshape(T, 1))
    simple_score = model.score(x_axis.reshape(T, 1), y_axis)
    print('score for 1 tree:', simple_score)

    # Bagging model
    B = 300
    bagging_model = BaggingRegressor(DecisionTreeRegressor, B)
    bagging_model.fit(Xtrain, Ytrain)
    bagging_prediction = bagging_model.predict(x_axis.reshape(T, 1))
    bagging_score = bagging_model.score(x_axis.reshape(T, 1), y_axis)
    print(f'bagging model score for {B} tree:{bagging_score}')

    plt.figure(figsize=(20, 10))
    plt.plot(x_axis, bagging_prediction, color='b', label=f'bagging model prediction (R2={bagging_score})')
    plt.plot(x_axis, simple_prediction, color='r', label=f'single model prediction (R2={simple_score})')
    plt.plot(x_axis, y_axis, color='y', label='true trend')
    plt.legend()
    plt.show()
