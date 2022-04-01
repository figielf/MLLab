import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary_2d(X, model):
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def plot_clusters(x, y, centres=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.7, s=20)

    if centres is not None:
        n_centres = len(centres)
        for i, c in enumerate(centres):
            plt.scatter(c[:, 0], c[:, 1], c=np.arange(c.shape[0]), marker='*', s=1000, alpha=(i + 1) / n_centres)
    plt.show()


def plot_clusters_history(x, clusters_hist, centres_hist=None):
    iters = len(clusters_hist)
    assert len(clusters_hist) == len(centres_hist)

    fig_n_cols = 4
    fig_n_rows = (iters // 4) + 1

    plt.figure(figsize=(8 * fig_n_cols, 8 * fig_n_rows))
    for i in range(iters):
        plt.subplot(fig_n_rows, fig_n_cols, i + 1)
        plt.scatter(x[:, 0], x[:, 1], c=clusters_hist[i], alpha=0.7, s=10)
        plt.scatter(centres_hist[i][:, 0], centres_hist[i][:, 1], c=np.arange(len(centres_hist[0])), marker='*', s=500)
        plt.title(f'iteration {i}')
    plt.show()


def plot_clusters_by_weights(x, weights, centres=None, color_base=None):
    n_clusters = weights.shape[1]
    if color_base is None:
        color_base = np.arange(n_clusters)

    plt.figure(figsize=(10, 10))
    plt.scatter(x[:, 0], x[:, 1], c=weights.dot(color_base), alpha=0.7, s=20)

    if centres is not None:
        n_centres = len(centres)
        for i, c in enumerate(centres):
            plt.scatter(c[:, 0], c[:, 1], c=color_base, marker='*', s=1000, alpha=(i + 1) / n_centres)
    plt.show()


def plot_clusters_by_weights_history(x, clusters_hist, weight_hist, centres_hist=None, color_base=None):
    iters = len(clusters_hist)
    assert len(clusters_hist) == len(centres_hist)

    fig_n_cols = 4
    fig_n_rows = (iters // 4) + 1

    n_clusters = weight_hist[-1].shape[1]
    if color_base is None:
        color_base = np.arange(n_clusters)

    plt.figure(figsize=(8 * fig_n_cols, 8 * fig_n_rows))
    for i in range(iters):
        plt.subplot(fig_n_rows, fig_n_cols, i + 1)
        plt.scatter(x[:, 0], x[:, 1], c=weight_hist[i].dot(color_base), alpha=0.7, s=20)
        plt.scatter(centres_hist[i][:, 0], centres_hist[i][:, 1], c=color_base, marker='*', s=1000)
        plt.title(f'iteration {i}')
    plt.show()
