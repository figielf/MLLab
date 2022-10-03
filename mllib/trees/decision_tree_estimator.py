import numpy as np
import pandas as pd
from scores import binary_entropy


class BinaryTreeNode:
    def __init__(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        self.depth = depth
        self.max_depth = max_depth
        self.max_bucket_size = max_bucket_size
        self.trace_logs = trace_logs
        if self.max_depth is not None and self.max_depth < self.depth:
            raise Exception(f'depth > max_depth:{depth > max_depth}, depth={depth}, max_depth={max_depth}')
        self.split_column_idx = None
        self.split_value = None
        self.left_child = None
        self.right_child = None
        self.prediction = None
        self.information_gain = None

    def fit(self, X, Y):
        if self.trace_logs:
            print(f'fit (depth:{self.depth}) - Start fit')
        if not self._is_fitted():
            if self._can_split(Y):
                split_column, self.split_value, self.information_gain = self._find_best_split(X, Y)
                if self.trace_logs:
                    print(f'fit (depth:{self.depth}) - (best_split_col, best_split_value, max_ig):'
                          f'{(split_column, self.split_value, self.information_gain)}')
                if split_column is None:
                    self.prediction = self._calc_prediction(Y)
                    return

                self.split_column_idx = int(split_column)
                left_split_mask = self._get_left_split_mask(X[:, self.split_column_idx], self.split_value)
                X_left, X_right = X[left_split_mask], X[~left_split_mask]
                Y_left, Y_right = Y[left_split_mask], Y[~left_split_mask]
                self.left_child = self._make_child_node(self.depth + 1, self.max_depth, self.max_bucket_size,
                                                        trace_logs=self.trace_logs)
                self.left_child.fit(X_left, Y_left)
                self.right_child = self._make_child_node(self.depth + 1, self.max_depth, self.max_bucket_size,
                                                         trace_logs=self.trace_logs)
                self.right_child.fit(X_right, Y_right)
            else:
                self.prediction = self._calc_prediction(Y)

    def predict(self, X):
        result = np.zeros(len(X))
        if self._is_leaf():
            return self.prediction
        left_split_mask = self._get_left_split_mask(X[:, self.split_column_idx], self.split_value)

        left_predictions = self.left_child.predict(X[left_split_mask])
        right_predictions = self.right_child.predict(X[~left_split_mask])

        result[left_split_mask] = left_predictions
        result[~left_split_mask] = right_predictions
        return result

    def get_importance(self):
        if not self._is_fitted():
            raise Exception(f'Node on level {self.depth} is not fitted yet')
        if self._is_leaf():  # no split no gain
            return np.array([(0, 0)])
        left_importance = self.left_child.get_importance()
        right_importance = self.right_child.get_importance()
        return self._calc_node_level_total_importance(left_importance, right_importance)

    def self_make_child_node(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        raise NotImplementedError()

    def _calc_prediction(self, y):
        raise NotImplementedError()

    def _calc_node_cost(self, y):
        raise NotImplementedError()

    def _calc_information_gain(self, y, split_mask):
        y0 = y[split_mask]
        y1 = y[~split_mask]
        N_0 = len(y0)
        N_1 = len(y1)
        N = N_0 + N_1
        if N_0 == 0 or N_1 == 0:
            return 0
        return self._calc_node_cost(y) - (N_0 * self._calc_node_cost(y0) + N_1 * self._calc_node_cost(y1)) / N

    def _calc_node_level_total_importance(self, left_child_importance, right_child_importance):
        importance = np.concatenate(
            (left_child_importance, right_child_importance, np.array([[self.split_column_idx, self.information_gain]])))
        importance_df = pd.DataFrame(importance, columns=['col_idx', 'information_gain'])
        return importance_df.groupby('col_idx', as_index=False).sum().values

    def _is_fitted(self):
        has_no_split_details = self.split_column_idx is None \
                               and self.split_value is None \
                               and self.left_child is None \
                               and self.right_child is None
        if self.prediction is None and has_no_split_details:  # not a leaf neither splitted
            return False
        if has_no_split_details and self.prediction is not None:  # is leaf
            return True
        has_split_details = self.split_column_idx is not None \
                            and self.split_value is not None \
                            and self.left_child is not None \
                            and self.right_child is not None
        if has_split_details and self.prediction is None:  # is splitted
            return True
        raise Exception(f'There are conflicting values in self.prediction and other attributes related to node split')

    def _is_leaf(self):
        if self._is_fitted():
            return self.prediction is not None
        return False

    def _can_split(self, y):
        # True if all below
        # 1. depth not bigger than allowed => self.depth <= self.max_depth
        # 2. num of obserwations bigger than requested => len(y) > self.max_bucket_size
        # 3. there is any variation in labels => (len(set(y)) == 1) > 1
        allowed = True
        if self.max_depth is not None:
            allowed = allowed and self.depth < self.max_depth
        if self.max_bucket_size is not None:
            allowed = allowed and len(y) > self.max_bucket_size
        if len(y) == 1 or len(set(y)) == 1:
            return False
        return allowed

    def _find_best_split(self, x, y):
        splits = self._get_split_candidates(x, y)
        if len(splits) == 0:
            return None, None, None
        return splits[np.argmax(splits[:, 2])]

    def _get_split_candidates(self, x, y):
        splits = []
        for i in range(x.shape[1]):
            x_col = x[:, i]
            if len(set(x_col)) == 1:
                continue
            sort_idx = np.argsort(x_col)
            x_col_sorted = x_col[sort_idx]
            y_sorted = y[sort_idx]
            steps_idx = self._get_steps(y_sorted)
            for s_idx in steps_idx:
                split_point = (x_col_sorted[s_idx] + x_col_sorted[s_idx + 1]) / 2.0
                left_split_mask = self._get_left_split_mask(x_col, split_point)
                ig = self._calc_information_gain(y, left_split_mask)
                if ig > 0.0:
                    splits.append([i, split_point, ig])
        return np.array(splits)

    def _get_left_split_mask(self, x, split_by_value):
        left_split_mask = x < split_by_value
        return left_split_mask

    def _get_steps(self, y):
        return np.nonzero(y[:-1] != y[1:])[0]


class BinaryTreeClassifierNode(BinaryTreeNode):
    def __init__(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        super().__init__(depth, max_depth, max_bucket_size, trace_logs)
        self.prediction_features = None

    def _calc_prediction(self, y):
        if len(y) == 1 or len(set(y)) == 1:
            return y[0]
        return int(np.round(y.mean()))

    def _calc_node_cost(self, y):
        return binary_entropy(y)

    def _make_child_node(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        return BinaryTreeClassifierNode(depth, max_depth, max_bucket_size, trace_logs)


class BinaryTreeRegressorNode(BinaryTreeNode):
    def __init__(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        super().__init__(depth, max_depth, max_bucket_size, trace_logs)
        self.prediction_features = None

    def _calc_prediction(self, y):
        if len(y) == 1 or len(set(y)) == 1:
            result = y[0]
        result = y.mean()
        if np.isnan(result):
            raise Exception(f'_calc_prediction calculated nan for the leaf, result={result}, len(y)={len(y)}')
        return result

    def _calc_node_cost(self, y):
        return np.var(y)

    def _make_child_node(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
        return BinaryTreeRegressorNode(depth, max_depth, max_bucket_size, trace_logs)


class BinaryTreeBase():
    def __init__(self, max_depth=10, max_bucket_size=10, trace_logs=True):
        self.max_depth = max_depth
        self.max_bucket_size = max_bucket_size
        self.trace_logs = trace_logs

    def predict(self, X):
        return self.head.predict(X)

    def score(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)

    def get_importance(self):
        return self.head.get_importance()


class BinaryTreeClassifier(BinaryTreeBase):
    def __init__(self, max_depth=10, max_bucket_size=10, trace_logs=True):
        super().__init__(max_depth, max_bucket_size, trace_logs)
        self.head = None

    def fit(self, X, Y):
        self.head = BinaryTreeClassifierNode(1, self.max_depth, self.max_bucket_size, trace_logs=self.trace_logs)
        self.head.fit(X, Y)


class BinaryTreeRegressor(BinaryTreeBase):
    def __init__(self, max_depth=10, max_bucket_size=10, trace_logs=True):
        super().__init__(max_depth, max_bucket_size, trace_logs)
        self.head = None

    def fit(self, X, Y):
        self.head = BinaryTreeRegressorNode(1, self.max_depth, self.max_bucket_size, trace_logs=self.trace_logs)
        self.head.fit(X, Y)

    def score(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)

    def get_importance(self):
        return self.head.get_importance()


class BinaryTreeClassifier(BinaryTreeBase):
    def __init__(self, max_depth=10, max_bucket_size=10, trace_logs=True):
        super().__init__(max_depth, max_bucket_size, trace_logs)
        self.head = None

    def fit(self, X, Y):
        self.head = BinaryTreeClassifierNode(1, self.max_depth, self.max_bucket_size, trace_logs=self.trace_logs)
        self.head.fit(X, Y)


class BinaryTreeRegressor(BinaryTreeBase):
    def __init__(self, max_depth=10, max_bucket_size=10, trace_logs=True):
        super().__init__(max_depth, max_bucket_size, trace_logs)
        self.head = None

    def fit(self, X, Y):
        self.head = BinaryTreeRegressorNode(1, self.max_depth, self.max_bucket_size, trace_logs=self.trace_logs)
        self.head.fit(X, Y)
