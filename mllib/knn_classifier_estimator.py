import numpy as np
import pandas as pd
from sortedcontainers import SortedList


def equides_metric(v1, v2):
    # print('v1:', v1)
    # print('v2:', v2)
    diff = (v1 - v2)
    # print('diff:', diff)
    return np.sqrt(diff.dot(diff))


class KnnClassifier:
    def __init__(self, k, distance_metric=equides_metric):
        self._k = k
        self._distance_metric = distance_metric

    def fit(self, X, Y):
        self._X = X
        self._Y = Y

    def predict(self, X):
        X = np.array(X)
        train_shape = self._X.shape[1:]
        predict_shape = X.shape[1:]

        if (train_shape != predict_shape):
            raise ValueError(
                f'Data X used for prediction are in incorect shape. Expected shape for each observation is {train_shape}, but was {predict_shape}.')

        X_dists = []
        for x in X:
            # print('x:', x)
            closest_neighbours = self._get_closest_neighbours(x)
            predicted_class = self._vote(closest_neighbours)
            X_dists.append(predicted_class)
        return np.array(X_dists)

    def _get_closest_neighbours(self, new_x):
        k_distances = SortedList()
        for i, x in enumerate(self._X):
            # print('i:', i)
            # print('x:', x)
            # print('new_x:', new_x)
            dist = self._distance_metric(new_x, x)

            # print(f'k_distances befor adding {dist}: {k_distances}')
            if (len(k_distances) < self._k):
                # print('(len(k_distances) < self._k)')
                k_distances.add((dist, self._Y[i]))
            else:
                last_dist = k_distances[-1][0]
                # print(f'else (last_dist:{last_dist}), (dist:{dist}), (diff last_dist-dist:{last_dist-dist})')
                if (dist == last_dist):
                    # print('(dist == k_distances[-1][0])')
                    k_distances.add((dist, self._Y[i]))
                if (dist < last_dist):
                    # print('(dist < k_distances[-1][0])')
                    k_distances.add((dist, self._Y[i]))

                    # print('k_distances:', k_distances)
                    k_th_dist, _ = k_distances[self._k - 1]
                    for i, (d, _) in enumerate(k_distances[self._k:]):
                        if (d > k_th_dist):
                            del k_distances[self._k + i:]
                            break
                    # print('k_th_dist:', k_th_dist)
            # print(f'k_distances after updating: {k_distances}')

        # print('k_distances:', k_distances)
        return k_distances

    def _vote(self, neighbours):
        # print('neighbours')
        # print(neighbours)
        df = pd.DataFrame(neighbours, columns=['distance', 'class'])
        # print('df:')
        # print(df)
        summary = df.groupby(by='class').agg({'class': 'count', 'distance': 'sum'}).rename(
            columns={'class': 'class_counts', 'distance': 'distance_sum'}).reset_index()
        # print('summary:')
        # print(summary)
        sorted_df = summary.sort_values(by=['class_counts', 'distance_sum'], ascending=[False, True])
        # print('sorted_df:')
        # print(sorted_df)
        best_classes = sorted_df['class'].values
        # print(type(sorted_df['class']))
        # print('winner_class:', best_classes[0])
        return best_classes[0]

    def score(self, X, Y):
        predictions = self.predict(X)
        # print('predictions:', predictions)
        # print('true Y:', Y)
        return np.mean(predictions == Y)