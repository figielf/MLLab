import sklearn.ensemble
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from random_forest_estimator import RandomForestRegressor
from model_selection import my_cross_val_score
from integration_tests.utils.data_utils import get_housing_data


if __name__ == '__main__':
    # Use Random Forest Regressor in housing data
    ESTIMATORS = 100
    X_train, X_test, Y_train, Y_test = get_housing_data(test_size=0.3)

    baseline = LinearRegression()
    single_tree = DecisionTreeRegressor()
    sklearn_rf = sklearn.ensemble.RandomForestRegressor(n_estimators=ESTIMATORS)
    my_rf_regressor = RandomForestRegressor(n_models=ESTIMATORS, n_features=5, trace_logs=False)

    single_tree.fit(X_train, Y_train)
    baseline.fit(X_train, Y_train)
    sklearn_rf.fit(X_train, Y_train)
    my_rf_regressor.fit(X_train, Y_train)

    single_tree_scores = my_cross_val_score(single_tree, X_train, Y_train, cv=5, shuffle=True, random_state=123)
    baseline_scores = my_cross_val_score(baseline, X_train, Y_train, cv=5, shuffle=True, random_state=123)
    sklearn_rf_scores = my_cross_val_score(sklearn_rf, X_train, Y_train, cv=5, shuffle=True, random_state=123)
    my_rf_regressor_scores = my_cross_val_score(my_rf_regressor, X_train, Y_train, cv=5, shuffle=True, random_state=123)

    print("test score single tree:", single_tree.score(X_test, Y_test))
    print("test score baseline:", baseline.score(X_test, Y_test))
    print("test sklearn score forest:", sklearn_rf.score(X_test, Y_test))
    print("test my score forest:", my_rf_regressor.score(X_test, Y_test))

    print("train score single tree:", single_tree.score(X_train, Y_train))
    print("train score baseline:", baseline.score(X_train, Y_train))
    print("train sklearn score forest:", sklearn_rf.score(X_train, Y_train))
    print("train my score forest:", my_rf_regressor.score(X_train, Y_train))

    print("CV single tree:", single_tree_scores.mean())
    print("CV baseline:", baseline_scores.mean())
    print("CV sklearn forest:", sklearn_rf_scores.mean())
    print("CV sklearn forest:", my_rf_regressor_scores.mean())
