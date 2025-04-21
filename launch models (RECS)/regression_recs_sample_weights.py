import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Regression & ML libraries
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import (
    BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# ----------- MODEL FUNCTIONS ----------- #

def regularization_fit(Model, X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table, efficiency):
    if not efficiency:
        alpha_ridge_max = 10
        alpha_lasso_min = 1
        alpha_lasso_max = 30
    else:
        alpha_ridge_max = 20
        alpha_lasso_min = 1e-10
        alpha_lasso_max = 10

    if Model == 'LinearRegression':
        model = LinearRegression()
        mod = 'Linear'
    elif Model == 'Ridge':
        alphas = np.linspace(0.0001, alpha_ridge_max, 100)
        model = RidgeCV(cv=None, alphas=alphas)
        mod = 'Ridge'
    elif Model == 'Lasso':
        alphas = np.linspace(alpha_lasso_min, alpha_lasso_max, 100)
        model = LassoCV(cv=None, alphas=alphas)
        mod = 'Lasso'
    else:
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=None, random_state=0)
        mod = 'ElasticNet'

    model.fit(X_train, y_train, sample_weight=weights_train)
    preds = model.predict(X_test)

    new_row = pd.DataFrame([{
        'Model': mod,
        'R2': r2_score(y_test, preds, sample_weight=weights_test),
        'MSE': mean_squared_error(y_test, preds, sample_weight=weights_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds, sample_weight=weights_test)),
        'MAE': mean_absolute_error(y_test, preds, sample_weight=weights_test),
        'MAPE': mean_absolute_percentage_error(y_test, preds, sample_weight=weights_test)
    }])

    summary_table = pd.concat([summary_table, new_row], ignore_index=True)

    if Model != 'LinearRegression':
        print(f"optimal alpha {Model}: {model.alpha_}")

    return summary_table, model

def ensemble_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table):
    rand_seed = 5
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X_train, y_train, weights_train, test_size=0.2, random_state=1
    )

    def log_and_save(name, model, X_val, y_val, X_test, y_test):
        y_pred = model.predict(X_test)
        scores = {
            'Model': name,
            'R2': r2_score(y_test, y_pred, sample_weight=weights_test),
            'MSE': mean_squared_error(y_test, y_pred, sample_weight=weights_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=weights_test)),
            'MAE': mean_absolute_error(y_test, y_pred, sample_weight=weights_test),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred, sample_weight=weights_test)
        }
        return pd.DataFrame([scores])

    # Bagging
    bag = BaggingRegressor(random_state=rand_seed, n_estimators=20, max_samples=100, max_features=10)
    bag.fit(X_train, y_train, sample_weight=weights_train)
    summary_table = pd.concat([summary_table, log_and_save('Bagging', bag, X_val, y_val, X_test, y_test)], ignore_index=True)

    # Random Forest
    rf = RandomForestRegressor(
        random_state=rand_seed, max_depth=15, n_estimators=20,
        max_samples=100, max_features='sqrt', min_samples_leaf=5
    )
    rf.fit(X_train, y_train, sample_weight=weights_train)
    summary_table = pd.concat([summary_table, log_and_save('RandomForest', rf, X_val, y_val, X_test, y_test)], ignore_index=True)

    # Gradient Boosting
    gb_grid = GridSearchCV(HistGradientBoostingRegressor(random_state=rand_seed), {'max_depth': range(1, 21)}, cv=5)
    gb_grid.fit(X_train, y_train, sample_weight=weights_train)
    gb = HistGradientBoostingRegressor(random_state=rand_seed, max_depth=gb_grid.best_params_['max_depth'])
    gb.fit(X_train, y_train, sample_weight=weights_train)
    summary_table = pd.concat([summary_table, log_and_save('Boosting', gb, X_val, y_val, X_test, y_test)], ignore_index=True)

    # AdaBoost
    ab_grid = GridSearchCV(AdaBoostRegressor(random_state=rand_seed), {'n_estimators': range(1, 21)}, cv=5)
    ab_grid.fit(X_train, y_train, sample_weight=weights_train)
    ab = AdaBoostRegressor(random_state=rand_seed, n_estimators=ab_grid.best_params_['n_estimators'])
    ab.fit(X_train, y_train, sample_weight=weights_train)
    summary_table = pd.concat([summary_table, log_and_save('AdaBoosting', ab, X_val, y_val, X_test, y_test)], ignore_index=True)

    return bag, rf, gb, ab, summary_table

def KNN_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table, splits):
    mses_knn = []
    for k in range(1, 31):
        knn = KNeighborsRegressor(n_neighbors=k)
        kfold = KFold(n_splits=splits, shuffle=True, random_state=1)
        scores = cross_val_score(knn, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        mses_knn.append(scores.mean())

    best_k = np.argmax(mses_knn) + 1
    knn = KNeighborsRegressor(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    row = pd.DataFrame([{
        'Model': 'KNN',
        'R2': r2_score(y_test, y_pred, sample_weight=weights_test),
        'MSE': mean_squared_error(y_test, y_pred, sample_weight=weights_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=weights_test)),
        'MAE': mean_absolute_error(y_test, y_pred, sample_weight=weights_test),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred, sample_weight=weights_test)
    }])
    summary_table = pd.concat([summary_table, row], ignore_index=True)

    return knn, np.sqrt(np.abs(mses_knn)), mses_knn, r2_score(y_test, y_pred), summary_table

def ANN_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table):
    param_space = {
        'hidden_layer_sizes': [(10, 30, 10), (20,), (64, 64, 64)],
        'activation': ['tanh', 'relu']
    }
    clf = GridSearchCV(MLPRegressor(max_iter=100), param_space, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    row = pd.DataFrame([{
        'Model': 'ANN',
        'R2': r2_score(y_test, y_pred, sample_weight=weights_test),
        'MSE': mean_squared_error(y_test, y_pred, sample_weight=weights_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=weights_test)),
        'MAE': mean_absolute_error(y_test, y_pred, sample_weight=weights_test),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred, sample_weight=weights_test)
    }])
    summary_table = pd.concat([summary_table, row], ignore_index=True)

    return clf, r2_score(y_test, y_pred), summary_table
