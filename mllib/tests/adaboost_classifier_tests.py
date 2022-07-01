import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from adaboost_classifier_estimator import AdaBoostClassifier
from tests.utils.data_utils import get_mushroom_data


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = get_mushroom_data(test_size=0.2)
    Y_train[Y_train == 0] = -1
    Y_test[Y_test == 0] = -1

    T = 200
    T_step = 10

    train_errors = []
    test_errors = []
    train_losses = []
    test_losses = []
    steps = []

    for num_trees in range(0, T + 1, T_step):
        if num_trees == 0:
            continue
        print('number of trees:', num_trees)

        model = AdaBoostClassifier(lambda: DecisionTreeClassifier(max_depth=1), n_steps=num_trees)
        model.fit(X_train, Y_train)
        acc, loss = model.score2(X_test, Y_test)
        acc_train, loss_train = model.score2(X_train, Y_train)
        train_errors.append(1 - acc_train)
        test_errors.append(1 - acc)
        train_losses.append(loss_train)
        test_losses.append(loss)
        steps.append(num_trees)

    plt.figure(figsize=(20, 10))
    plt.plot(steps, test_errors, label='test errors')
    plt.plot(steps, test_losses, label='test losses')
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(steps, train_errors, label='train errors')
    plt.plot(steps, train_losses, label='test losses')
    plt.legend()
    plt.show()

    print("final train error:", 1 - acc_train)
    print("final train loss:", loss_train)
    print("final test error:", 1 - acc)
    print("final testloss:", loss)
