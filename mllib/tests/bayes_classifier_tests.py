from matplotlib import pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

from bayesian.bayes_classifier_generic import bayes_classifier
from tests.utils.data_utils import get_mnist_data

if __name__ == '__main__':
    K = 10
    N = 1000
    X, _, Y, _, picture_shape = get_mnist_data(train_size=1.0, should_plot_examples=False)
    X = X[:200]
    Y = Y[:200]

    clf = bayes_classifier(n_classes=K, model_factory=lambda: BayesianGaussianMixture(n_components=10))
    clf.fit(X, Y)

    plt.figure(figsize=(16, 16))
    for k in range(clf.K):
        sample, sample_class = clf.sample_given_y(k)
        sample_class_mean = clf.models[k].means_[sample_class]  # mean on sample_class-th gaussian withing k-th gmm

        plt.subplot(5, 4, 2 * k + 1)
        plt.imshow(sample.reshape(picture_shape), cmap='gray')
        plt.title(f'Sample of {k}')
        plt.subplot(5, 4, 2 * k + 2)
        plt.imshow(sample_class_mean.reshape(picture_shape), cmap='gray')
        plt.title(f'Mean of {k}')
    plt.show()

    # generate a random sample
    (sample, sample_class), digit = clf.sample()
    sample_class_mean = clf.models[digit].means_[sample_class]

    plt.subplot(1, 2, 1)
    plt.imshow(sample.reshape(picture_shape), cmap='gray')
    plt.title('Random Sample from Random Class')
    plt.subplot(1, 2, 2)
    plt.imshow(sample_class_mean.reshape(picture_shape), cmap='gray')
    plt.title('Corresponding Cluster Mean')
    plt.show()
