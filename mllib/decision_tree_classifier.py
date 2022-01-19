import numpy as np
import pandas as pd


def binary_entropy(y):
  # assume y is binary - 0 or 1
  N = len(y)
  s1 = (y == 1).sum()
  if 0 == s1 or N == s1:
    return 0
  p1 = float(s1) / N
  p0 = 1 - p1
  return -p0 * np.log2(p0) - p1 * np.log2(p1)


class BinaryTreeNode:
  split_column_idx = None
  split_value = None
  left_child = None
  right_child = None
  prediction = None
  information_gain = None

  def __init__(self, depth, max_depth=None, max_bucket_size=None, trace_logs=True):
    # print(f'Creating new node with depth={depth}')
    self.depth = depth
    self.max_depth = max_depth
    self.max_bucket_size = max_bucket_size
    self.trace_logs = trace_logs
    if self.max_depth is not None and self.max_depth < self.depth:
      raise Exception(f'depth > max_depth:{depth > max_depth}, depth={depth}, max_depth={max_depth}')

  def fit(self, X, Y):
    if (self.trace_logs == True):
      print(f'fit (depth:{self.depth}) - Start fit')
      # print(f'fit (depth:{self.depth}) - X:{X}')
      # print(f'fit (depth:{self.depth}) - Y:{Y}')
    if (self._is_fitted() == False):
      if (self._can_split(Y)):
        # print('Is allowed to split')
        split_column, self.split_value, self.information_gain = self._find_best_split(X, Y)
        if (self.trace_logs == True):
          print(
            f'fit (depth:{self.depth}) - (best_split_col, best_split_value, max_ig):{(split_column, self.split_value, self.information_gain)}')
        if (split_column is None):
          print('fit (depth:{self.depth}) - no splits found, will make this node a leaf')
          self.prediction = self._calc_prediction(Y)
          # print(f'fit (depth:{self.depth}) - Leaf on level {self.depth}, calculated prediction={self.prediction}')
          return

        self.split_column_idx = int(split_column)
        left_split_mask = self._get_left_split_mask(X[:, self.split_column_idx], self.split_value)
        X_left, X_right = X[left_split_mask], X[~left_split_mask]
        Y_left, Y_right = Y[left_split_mask], Y[~left_split_mask]
        # print(f'fit (depth:{self.depth}) - Y_left len:{len(Y_left)}, Y_right len:{len(Y_right)}')
        # print(f'fit (depth:{self.depth}) - Y before split len:{len(Y)}, Y:{Y}')
        self.left_child = BinaryTreeNode(self.depth + 1, self.max_depth, self.max_bucket_size,
                                         trace_logs=self.trace_logs)
        self.left_child.fit(X_left, Y_left)
        self.right_child = BinaryTreeNode(self.depth + 1, self.max_depth, self.max_bucket_size,
                                          trace_logs=self.trace_logs)
        self.right_child.fit(X_right, Y_right)
      else:
        # print('fit (depth:{self.depth}) - Is not allowed to split')
        # print('fit (depth:{self.depth}) - Y:', Y)
        self.prediction = self._calc_prediction(Y)
        # print(f'fit (depth:{self.depth}) - Calculated prediction={self.prediction}')

  def predict(self, X):
    result = np.zeros(len(X))
    # print(f'predict - node level={self.depth}')
    if (self._is_leaf() == True):
      # print('Node has not childrens')
      # print('predict - self.prediction:', self.prediction)
      return self.prediction
    left_split_mask = self._get_left_split_mask(X[:, self.split_column_idx], self.split_value)

    left_predictions = self.left_child.predict(X[left_split_mask])
    right_predictions = self.right_child.predict(X[~left_split_mask])
    # print(f'predict - left_predictions:{left_predictions}, left_split_mask:{left_split_mask}')
    # print(f'predict - right_predictions:{right_predictions}, right_split_mask:{~left_split_mask}')

    # print('predict - left_split_mask:', left_split_mask)
    result[left_split_mask] = left_predictions
    result[~left_split_mask] = right_predictions
    # print('predict - result:', result)
    return result

  def get_importance(self):
    # tabs = '\t'*self.depth
    # print(f'{tabs}get_importance - node level {self.depth}')
    if (self._is_fitted() == False):
      raise Exception(f'Node on level {self.depth} is not fitted yet')
    if (self._is_leaf()):  # no split no gain
      return np.array([(0, 0)])
    left_importance = self.left_child.get_importance()
    # print(f'{tabs}get_importance on level {self.depth} - left_importance:{left_importance}')
    right_importance = self.right_child.get_importance()
    # print(f'{tabs}get_importance on level {self.depth} - right_importance:{right_importance}')
    return self._calc_node_level_total_importance(left_importance, right_importance)

  def _calc_node_level_total_importance(self, left_child_importance, right_child_importance):
    tabs = '\t' * self.depth
    importances = np.concatenate(
      (left_child_importance, right_child_importance, np.array([[self.split_column_idx, self.information_gain]])))
    # print(f'{tabs}_calc_node_level_total_importance on level {self.depth} - not summed importances:')
    # print(importances)
    importances_df = pd.DataFrame(importances, columns=['col_idx', 'information_gain'])
    importance = importances_df.groupby('col_idx', as_index=False).sum().values
    # print(f'{tabs}_calc_node_level_total_importance on level {self.depth} - summed importance:')
    # print(importance)
    return importance

  def _is_fitted(self):
    has_no_split_details = self.split_column_idx is None and self.split_value is None and self.left_child is None and self.right_child is None
    if (self.prediction is None and has_no_split_details):  # not a leaf neither splitted
      return False
    if (self.prediction is not None and has_no_split_details == True):  # is leaf
      return True
    has_split_details = self.split_column_idx is not None and self.split_value is not None and self.left_child is not None and self.right_child is not None
    if (self.prediction is None and has_split_details == True):  # is splitted
      return True
    raise Exception(f'There are conflicting values in self.prediction and other attributes related to node split')

  def _is_leaf(self):
    if (self._is_fitted()):
      return self.prediction is not None
    return False

  def _can_split(self, Y):
    # True if all below
    # 1. depth not bigger than allowed => self.depth <= self.max_depth
    # 2. num of obserwations bigger than requested => len(Y) > self.max_bucket_size
    # 3. there is any variation in labels => (len(set(Y)) == 1) > 1
    allowed = True
    # print('should_try_split - result:', allowed)
    if (self.max_depth is not None):
      # print(f'self.max_depth is not None and node.depth <= self.max_depth={node.depth <= self.max_depth}, node.depth={node.depth}, self.max_depth={self.max_depth}')
      allowed = allowed and self.depth < self.max_depth
    # print('should_try_split - result:', allowed)
    if (self.max_bucket_size is not None):
      # print(f'self.max_bucket_size is not None and bucket_size > self.max_bucket_size={bucket_size > self.max_bucket_size}, node.bucket_size={bucket_size}, self.max_bucket_size={self.max_bucket_size}')
      allowed = allowed and len(Y) > self.max_bucket_size
    if (len(Y) == 1 or len(set(Y)) == 1):
      return False
    # print('should_try_split - result:', allowed)
    return allowed

  def _find_best_split(self, X, Y):
    splits = self._get_split_candidates(X, Y)
    # print('_find_best_split - splits:', pd.DataFrame(splits, columns=['column_idx', 'split_value', 'ig']))
    if (len(splits) == 0):
      return (None, None, None)
    return splits[np.argmax(splits[:, 2])]

  def _get_split_candidates(self, X, Y):
    splits = []
    for i in range(X.shape[1]):
      X_col = X[:, i]
      if (len(set(X_col)) == 1):
        # print(f'_find_all_splits - all split column {i} valueas are same (={X[:,i][0]}) and should no split further')
        continue
      sort_idx = np.argsort(X_col)
      X_col_sorted = X_col[sort_idx]
      Y_sorted = Y[sort_idx]
      steps_idx = self._get_steps(Y_sorted)
      # print(f'_find_all_splits - column={i}, steps_idx:{steps_idx}, X_col_sorted:{X_col_sorted}, Y_sorted:{Y_sorted}')
      for s_idx in steps_idx:
        split_point = (X_col_sorted[s_idx] + X_col_sorted[s_idx + 1]) / 2.0
        # print('_find_all_splits - split_point:', split_point)
        left_split_mask = self._get_left_split_mask(X_col, split_point)
        ig = self._calc_information_gain(Y[left_split_mask], Y[~left_split_mask])
        # print('_find_all_splits - calculated information gain:', ig)
        splits.append([i, split_point, ig])
    return np.array(splits)

  def _calc_prediction(self, Y):
    if (len(Y) == 1 or len(set(Y)) == 1):
      return Y[0]
    return int(np.round(Y.mean()))

  def _get_left_split_mask(self, x, split_by_value):
    # print(f'_get_left_split_mask - split_by_value:{split_by_value}')
    # print(f'_get_left_split_mask - x:{x}')
    left_split_mask = x < split_by_value
    # print('_get_split_mask - left_split_mask:', left_split_mask)
    return left_split_mask

  def _calc_information_gain(self, y0, y1):
    # print(f'_calc_information_gain - y0:{y0}')
    # print(f'_calc_information_gain - y1:{y1}')
    N_0 = len(y0)
    N_1 = len(y1)
    N = N_0 + N_1
    if (N_0 == 0 or N_1 == 0):
      # print(f'_calc_information_gain - one leg (left:{N_0}, right:{N_1}) is lenght of 0 so 0 is returned as information gain')
      return 0
    y = np.concatenate((y0, y1))
    # print(f'_calc_information_gain - concatenation of left:{y0}, right:{y1}, result:{y}')
    # print(f'_calc_information_gain - binary_entropy(y):{binary_entropy(y)}, binary_entropy(y0):{binary_entropy(y0)}, binary_entropy(y1):{binary_entropy(y1)}')
    return binary_entropy(y) - (N_0 * binary_entropy(y0) + N_1 * binary_entropy(y1)) / N

  def _get_steps(self, Y):
    return np.nonzero(Y[:-1] != Y[1:])[0]


class BinaryTreeClassifier:
  def __init__(self, max_depth=10, max_bucket_size=10, trace_logs=True):
    self.max_depth = max_depth
    self.max_bucket_size = max_bucket_size
    self.trace_logs = trace_logs

  def fit(self, X, Y):
    self.head = BinaryTreeNode(1, self.max_depth, self.max_bucket_size, trace_logs=self.trace_logs)
    self.head.fit(X, Y)

  def predict(self, X):
    return self.head.predict(X)

  def score(self, X, Y):
    predictions = self.predict(X)
    return np.mean(predictions == Y)

  def get_importance(self):
    return self.head.get_importance()