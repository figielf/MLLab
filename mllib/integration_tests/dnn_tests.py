from integration_tests.utils.data_utils import get_mnist_data

if __name__ == '__main__':
    K = 10
    test_size = 1000
    Xtrain, Xtest, Ytrain, Ytest, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]
    Xtest, Ytest = Xtest[-test_size:], Ytest[-test_size:]

    # dnn = DNN([1000, 750, 500])
    # dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3)
    # vs
    dnn = dnn([1000, 750, 500])
    dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=True, train_head_only=False, epochs=3)
    # note: try training the head only too! what does that mean?