import numpy as np
import sklearn.ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from random_forest_estimator import RandomForestRegressor
from model_selection import my_cross_val_score
from tests.utils.data_utils import get_mushroom_data


if __name__ == '__main__':
    ESTIMATORS = 10
    X_train, X_test, Y_train, Y_test = get_mushroom_data(test_size=0.3)

    baseline = LogisticRegression()
    single_tree = DecisionTreeClassifier()
    sklearn_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=ESTIMATORS)
    my_rf_classifier = RandomForestRegressor(n_models=ESTIMATORS,
                                             n_features=int(np.sqrt(X_train.shape[1])),
                                             trace_logs=False)

    single_tree.fit(X_train, Y_train)
    baseline.fit(X_train, Y_train)
    sklearn_rf.fit(X_train, Y_train)
    my_rf_classifier.fit(X_train, Y_train)

    single_tree_scores = my_cross_val_score(single_tree, X_train, Y_train, cv=5, shuffle=True, random_state=123)
    baseline_scores = my_cross_val_score(baseline, X_train, Y_train, cv=5, shuffle=True, random_state=123)
    sklearn_rf_scores = my_cross_val_score(sklearn_rf, X_train, Y_train, cv=5, shuffle=True, random_state=123)
    my_rf_classifier_scores = my_cross_val_score(my_rf_classifier, X_train, Y_train, cv=5, shuffle=True,
                                                 random_state=123)

    print("test score single tree:", single_tree.score(X_test, Y_test))
    print("test score baseline:", baseline.score(X_test, Y_test))
    print("test sklearn score forest:", sklearn_rf.score(X_test, Y_test))
    print("test my score forest:", my_rf_classifier.score(X_test, Y_test))

    print("train score single tree:", single_tree.score(X_train, Y_train))
    print("train score baseline:", baseline.score(X_train, Y_train))
    print("train sklearn score forest:", sklearn_rf.score(X_train, Y_train))
    print("train my score forest:", my_rf_classifier.score(X_train, Y_train))

    print("CV single tree:", single_tree_scores.mean())
    print("CV baseline:", baseline_scores.mean())
    print("CV sklearn forest:", sklearn_rf_scores.mean())
    print("CV sklearn forest:", my_rf_classifier_scores.mean())