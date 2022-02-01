import numpy as np
import pandas as pd
from sortedcontainers import SortedList


def equides_metric(v1, v2):
    diff = (v1 - v2)
    return np.sqrt(diff.dot(diff))


class KnnClassifier:
    def __init__(self, k, distance_metric=equides_metric):
        self._k = k
        self._distance_metric = distance_metric
        self._X = None
        self._Y = None

    def fit(self, X, Y):
        self._X = X
        self._Y = Y

    def predict(self, X):
        X = np.array(X)
        train_shape = self._X.shape[1:]
        predict_shape = X.shape[1:]

        if train_shape != predict_shape:
            raise ValueError(
                f'Data X used for prediction are in incorect shape. Expected shape for each observation is '
                f'{train_shape}, but was {predict_shape}.')

        X_dists = []
        for x in X:
            closest_neighbours = self._get_closest_neighbours(x)
            predicted_class = self._vote(closest_neighbours)
            X_dists.append(predicted_class)
        return np.array(X_dists)

    def _get_closest_neighbours(self, new_x):
        k_distances = SortedList()
        for i, x in enumerate(self._X):
            dist = self._distance_metric(new_x, x)

            if len(k_distances) < self._k:
                k_distances.add((dist, self._Y[i]))
            else:
                last_dist = k_distances[-1][0]
                if dist == last_dist:
                    k_distances.add((dist, self._Y[i]))
                if dist < last_dist:
                    k_distances.add((dist, self._Y[i]))

                    k_th_dist, _ = k_distances[self._k - 1]
                    for j, (d, _) in enumerate(k_distances[self._k:]):
                        if d > k_th_dist:
                            del k_distances[self._k + j:]
                            break
        return k_distances

    @staticmethod
    def _vote(neighbours):
        df = pd.DataFrame(neighbours, columns=['distance', 'class'])
        summary = df.groupby(by='class').agg({'class': 'count', 'distance': 'sum'}).rename(
            columns={'class': 'class_counts', 'distance': 'distance_sum'}).reset_index()
        sorted_df = summary.sort_values(by=['class_counts', 'distance_sum'], ascending=[False, True])
        best_classes = sorted_df['class'].values
        return best_classes[0]

    def score(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)
