import numpy as np
from decision_tree_estimator import BinaryTreeNode, BinaryTreeBase
from bagging_estimator import BaggingRegressor, BaggingClassifier
from scores import binary_entropy


class RandomForestBinaryTreeNode(BinaryTreeNode):
    def __init__(self, depth, max_depth=None, max_bucket_size=None, n_features=None, trace_logs=True):
        super().__init__(depth, max_depth, max_bucket_size, trace_logs)
        self.n_features = n_features
        self.split_features_subset = None

    def _find_best_split(self, x, y):
        if self.n_features is not None:
            if self.n_features > x.shape[1]:
                raise Exception(
                    f'n_features={self.n_features} can not be bigger than number of features in X={x.shape[1]}')
            self.split_features_subset = np.sort(np.random.choice(x.shape[1], size=self.n_features, replace=False))
            best_split = super()._find_best_split(x[:, self.split_features_subset], y)
            # print(f'_find_best_split - type(best_split):{type(best_split)}, best_split:{best_split}')
            best_split_col_idx, best_split_point, best_ig = best_split[0], best_split[1], best_split[2]
            if best_split_col_idx is None:
                return None, None, None
            return [self.split_features_subset[int(best_split_col_idx)], best_split_point, best_ig]
        else:
            return super()._find_best_split(x, y)


class RandomForestTreeClassifierNode(RandomForestBinaryTreeNode):
    def __init__(self, depth, max_depth=None, max_bucket_size=None, n_features=None, trace_logs=True):
        super().__init__(depth, max_depth, max_bucket_size, n_features, trace_logs)

    def _calc_prediction(self, y):
        if len(y) == 1 or len(set(y)) == 1:
            result = y[0]
        result = int(np.round(y.mean()))
        if np.isnan(result):
            raise Exception(f'_calc_prediction calculated nan for the leaf, result={result}, len(y)={len(y)}')
        return result

    def _calc_node_cost(self, y):
        return binary_entropy(y)

    def _make_child_node(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        return RandomForestTreeClassifierNode(depth, max_depth, max_bucket_size, self.n_features, trace_logs)


class RandomForestTreeRegressorNode(RandomForestBinaryTreeNode):
    def __init__(self, depth, max_depth=None, max_bucket_size=None, n_features=None, trace_logs=True):
        self.n_features = n_features
        super().__init__(depth, max_depth, max_bucket_size, n_features, trace_logs)

    def _calc_prediction(self, y):
        if len(y) == 1 or len(set(y)) == 1:
            result = y[0]
        result = y.mean()
        if np.isnan(result):
            raise Exception(f'_calc_prediction calculated nan for the leaf, result={result}, len(y)={len(y)}')
        return result

    def _calc_node_cost(self, y):
        if y is None or len(y) <= 1:
            return 0
        return np.var(y)

    def _make_child_node(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        return RandomForestTreeRegressorNode(depth, max_depth, max_bucket_size, self.n_features, trace_logs)


class BinaryTreeForRandomForestClassifier(BinaryTreeBase):
    def __init__(self, max_depth=10, max_bucket_size=10, n_features=None, trace_logs=True):
        super().__init__(max_depth, max_bucket_size, trace_logs)
        self.n_features = n_features
        self.head = None

    def fit(self, X, Y):
        self.head = RandomForestTreeClassifierNode(1, self.max_depth, self.max_bucket_size, self.n_features,
                                                   trace_logs=self.trace_logs)
        self.head.fit(X, Y)


class BinaryTreeForRandomForestRegressor(BinaryTreeBase):
    def __init__(self, max_depth=10, max_bucket_size=10, n_features=None, trace_logs=True):
        super().__init__(max_depth, max_bucket_size, trace_logs)
        self.n_features = n_features
        self.head = None

    def fit(self, X, Y):
        self.head = RandomForestTreeRegressorNode(1, self.max_depth, self.max_bucket_size, self.n_features,
                                                  trace_logs=self.trace_logs)
        self.head.fit(X, Y)


class RandomForestClassifier:
    def __init__(self, n_models, sample_size=None, n_features=None, max_depth=None, max_bucket_size=None,
                 trace_logs=True):
        self.n_models = n_models
        self.sample_size = sample_size
        self.n_features = n_features
        self.max_depth = max_depth
        self.max_bucket_size = max_bucket_size
        self.trace_logs = trace_logs
        self.bagged_tree = None

    def fit(self, X, Y):
        self.bagged_tree = BaggingClassifier(lambda: BinaryTreeForRandomForestClassifier(
                                                        self.max_depth,
                                                        self.max_bucket_size,
                                                        self.n_features,
                                                        self.trace_logs),
                                             self.n_models,
                                             self.sample_size)
        self.bagged_tree.fit(X, Y)

    def predict(self, X):
        predictions = self.bagged_tree.predict(X)
        return predictions

    def score(self, X, Y):
        return self.bagged_tree.score(X, Y)


class RandomForestRegressor:
    def __init__(self, n_models,
                 sample_size=None,
                 n_features=None,
                 max_depth=None,
                 max_bucket_size=None,
                 trace_logs=True):
        self.n_models = n_models
        self.sample_size = sample_size
        self.n_features = n_features
        self.max_depth = max_depth
        self.max_bucket_size = max_bucket_size
        self.trace_logs = trace_logs
        self.bagged_tree = None

    def fit(self, X, Y):
        self.bagged_tree = BaggingRegressor(lambda: BinaryTreeForRandomForestRegressor(
                                                        self.max_depth,
                                                        self.max_bucket_size,
                                                        self.n_features,
                                                        self.trace_logs),
                                            self.n_models,
                                            self.sample_size)
        self.bagged_tree.fit(X, Y)

    def predict(self, X):
        predictions = self.bagged_tree.predict(X)
        return predictions

    def score(self, X, Y):
        return self.bagged_tree.score(X, Y)
