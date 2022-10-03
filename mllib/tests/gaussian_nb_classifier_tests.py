from sklearn.decomposition import PCA

from dimensionality_reduction.pca import pca
from bayesian.gaussian_nb_with_pca_classifier import gaussian_nb_with_pca_classifier
from bayesian.gaussian_nb_classifier import gaussian_nb_classifier
from tests.utils.data_utils import get_mnist_data


def test_nb_model(Xtrain, Ytrain, Xtest, Ytest, model_factory, info):
    # try NB by itself
    model1 = model_factory()
    model1.fit(Xtrain, Ytrain)
    print(info + " - NB train score:", model1.score(Xtrain, Ytrain))
    print(info + " - NB test score:", model1.score(Xtest, Ytest))

    # try NB with PCA first
    pca_model = PCA(n_components=50)
    Ztrain = pca_model.fit_transform(Xtrain)
    Ztest = pca_model.transform(Xtest)

    model2 = model_factory()
    model2.fit(Ztrain, Ytrain)
    print(info + " - NB+PCA train score:", model2.score(Ztrain, Ytrain))
    print(info + " - NB+PCA test score:", model2.score(Ztest, Ytest))

    # try NB with my PCA first
    pca_model = pca(n_components=50)
    Ztrain = pca_model.fit_transform(Xtrain)
    Ztest = pca_model.transform(Xtest)

    model2 = model_factory()
    model2.fit(Ztrain, Ytrain)
    print(info + " - NB+my PCA train score:", model2.score(Ztrain, Ytrain))
    print(info + " - NB+my PCA test score:", model2.score(Ztest, Ytest))


if __name__ == '__main__':
    K = 10
    N = 1000
    X_train, X_test, Y_train, Y_test, picture_shape = get_mnist_data(train_size=0.8, should_plot_examples=False)

    test_nb_model(X_train, Y_train, X_test, Y_test, lambda: gaussian_nb_classifier(K), 'gaussian_nb_classifier')

    # try NB by itself
    model1 = gaussian_nb_with_pca_classifier(n_classes=K, n_components=50)
    model1.fit(X_train, Y_train)
    print("gaussian_nb_with_pca_classifier - NB train score:", model1.score(X_train, Y_train))
    print("gaussian_nb_with_pca_classifier - NB test score:", model1.score(X_test, Y_test))
