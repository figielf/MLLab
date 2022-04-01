import numpy as np


# purity clustering measure
def purity_cost(y, clusters):
    n_clusters = len(set(y))
    clusters_purity = []
    for c1 in range(n_clusters):
        one_class_putiry = []
        for c2 in range(n_clusters):
            one_class_putiry.append(np.sum((clusters == c1) & (y == c2)))
        clusters_purity.append(np.max(one_class_putiry))
    return np.sum(clusters_purity) / len(y)


def purity_soft_cost(y, weights):
    n_clusters = weights.shape[1]
    clusters_purity = []
    for c1 in range(n_clusters):
        one_class_putiry = []
        for c2 in range(n_clusters):
            one_class_putiry.append(np.sum(weights[y == c2, c1]))
        clusters_purity.append(np.max(one_class_putiry))
    return np.sum(clusters_purity) / len(y)


def calc_purity_hist(y, clusters_hist):
    purity_hist = []
    for c in clusters_hist:
        purity_hist.append(purity_cost(y, c))
    return purity_hist


def calc_purity_soft_hist(y, weights_hist):
    putiry_hist = []
    for r in weights_hist:
        putiry_hist.append(purity_soft_cost(y, r))
    return putiry_hist


# davis-bouldin index clustering measure

def davis_bouldin_index_cost(x, clusters, cluster_centres, dist_calc_func):
    K = len(cluster_centres)
    sigma = np.zeros(K)
    for k in range(K):
        d = dist_calc_func(x[clusters == k] - cluster_centres[k])
        sigma[k] = np.sqrt(d.mean())

    clusters_divergence = []
    for i in range(K):
        one_class_divergence = []
        for j in range(K):
            if i != j:
                dc = np.sqrt(dist_calc_func(cluster_centres[i] - cluster_centres[j]))
                one_class_divergence.append((sigma[i] + sigma[j]) / dc)
        clusters_divergence.append(np.max(one_class_divergence))
    return np.mean(clusters_divergence)


def davis_bouldin_index_soft_cost(x, weights, cluster_centres, dist_calc_func):
    K = len(cluster_centres)
    sigma = np.zeros(K)
    for k in range(K):
        d = weights[:, k] * dist_calc_func(x - cluster_centres[k])
        sigma[k] = np.sqrt(d.sum() / weights[:, k].sum())

    clusters_divergence = []
    for i in range(K):
        one_class_divergence = []
        for j in range(K):
            if i != j:
                dc = np.sqrt(dist_calc_func(cluster_centres[i] - cluster_centres[j]))
                one_class_divergence.append((sigma[i] + sigma[j]) / dc)
        clusters_divergence.append(np.max(one_class_divergence))
    return np.mean(clusters_divergence)


def calc_davis_bouldin_index_hist(x, clusters_hist, centres_hist, dist_calc_func):
    index_hist = []
    for i in range(len(clusters_hist)):
        index_hist.append(davis_bouldin_index_cost(x, clusters_hist[i], centres_hist[i], dist_calc_func))
    return index_hist


def calc_davis_bouldin_index_soft_hist(x, weights_hist, centres_hist, dist_calc_func):
    index_hist = []
    for i in range(len(weights_hist)):
        index_hist.append(davis_bouldin_index_soft_cost(x, weights_hist[i], centres_hist[i], dist_calc_func))
    return index_hist
