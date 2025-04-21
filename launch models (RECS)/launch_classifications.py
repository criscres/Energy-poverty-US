import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
import geopandas as gpd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Dependencies from our .py files
import recs_EDA
import classification_recs_sample_weights as classif
import compare_classification_models as compare_class

warnings.filterwarnings('ignore')


# --- Clean VBS ---
def clean_vbs(RECS_raw):
    """
    Clean and transform the RECS dataset.
    """
    X_sub, Y, RECS_norm_param = recs_EDA.vb_transform(RECS_raw)
    return X_sub, Y


# --- Bootstrap Weights and Feature Engineering ---
def bootstrap_weights_feature_eng(X_sub, y, feature_engineer=True):
    """
    Perform train-test split and optionally apply feature engineering.
    """
    test_prop = 0.2
    b = 0

    # Drop weight column and split data
    weights = X_sub.NWEIGHT
    X = X_sub.drop('NWEIGHT', axis=1)
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, weights, test_size=test_prop, random_state=b
    )

    # Feature Engineering
    if feature_engineer:
        poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        X_train_eng = poly.fit_transform(X_train)
        X_test_eng = poly.transform(X_test)

        X_train = pd.DataFrame(X_train_eng, columns=poly.get_feature_names_out(X_train.columns))
        X_test = pd.DataFrame(X_test_eng, columns=poly.get_feature_names_out(X_test.columns))

    return X_train, X_test, y_train, y_test, sw_train, sw_test


# --- Run Models ---
def run_models(summary_table, save_path, thresh):
    """
    Run classification models and update the summary table.
    """
    # Logistic Regression
    logreg, summary_table, y_pred_logistic = classif.logistic_fit(
        X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, thresh
    )

    # KNN with CV
    splits = 5
    knn_cv, summary_table, y_pred_KNN = classif.KNN_fit(
        X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, splits, thresh
    )

    # ANN
    clf, summary_table, y_pred_ANN = classif.ANN_fit(
        X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, thresh
    )

    # Ensemble Methods
    bag_tree, rf_tree, gb_tree, ab_tree, summary_table, y_pred_bag, y_pred_rf, y_pred_gb, y_pred_ab = classif.ensemble_fit(
        X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, thresh
    )

    # SVC
    grid, summary_table, y_pred_svc = classif.SVC_fit(
        X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, thresh
    )

    # Compare Models
    class_metrics = comparison_roc(
        y_pred_logistic, y_pred_KNN, y_pred_ANN, y_pred_bag, y_pred_rf, y_pred_gb, y_pred_ab, y_pred_svc, save_path, thresh
    )

    return class_metrics


# --- Compare Models and Plot ROC ---
def comparison_roc(y_pred_logistic, y_pred_KNN, y_pred_ANN, y_pred_bag, y_pred_rf, y_pred_gb, y_pred_ab, y_pred_svc, save_path, thresh):
    """
    Compare models, plot ROC curves, and save AUC results.
    """
    list_models_prob = {
        'log_model': y_pred_logistic,
        'KNN_model': y_pred_KNN,
        'ANN_model': y_pred_ANN,
        'bag_model': y_pred_bag,
        'rf_model': y_pred_rf,
        'gb_model': y_pred_gb,
        'ab_model': y_pred_ab,
        'svc_model': y_pred_svc
    }

    # Create class_metrics DataFrame with additional columns for AUC
    class_metrics = pd.DataFrame(
        columns=['Model', 'matthews corr', 'accuracy score', 'f1', 'precision score', 'recall score', 'Kappa', 'AUC_ROC', 'AUC_PR']
    )

    for key, value in list_models_prob.items():
        y_score = value[:, 1]
        savetitle = y_name + '_' + key
        
        # Generate plots and obtain metrics including AUC values
        plotter = compare_class.PlotAll(savetitle, savetitle, y_test, y_score, save_path, thresh)
        matthews, acc, f1, prec, rec, kap, auc_roc, auc_pr = plotter.get_metrics()

        # Create new row with model metrics
        new_row = pd.DataFrame({
            'Model': [key],
            'matthews corr': [matthews],
            'accuracy score': [acc],
            'f1': [f1],
            'precision score': [prec],
            'recall score': [rec],
            'Kappa': [kap],
            'AUC_ROC': [auc_roc],
            'AUC_PR': [auc_pr]
        })

        # Use pd.concat to add to class_metrics
        class_metrics = pd.concat([class_metrics, new_row], ignore_index=True)

    return class_metrics


# --- Main Execution ---
if __name__ == "__main__":

    # Import RECS Data
    path = '/global/scratch/users/cristina_crespo/p1_data/'
    RECS_raw = pd.read_csv(path + 'RECS/recs2020_public_v5.csv')

    # Select dependent variable
    y_name = 'SCALEE'  # Options: 'SCALEG', 'SCALEB', 'ENERGYASST', 'SCALEE'
    print(y_name)

    # Set up summary table to compare models
    summary_table = pd.DataFrame(columns=['Model', 'score'])

    # Clean RECS data
    X_sub, Y = clean_vbs(RECS_raw)

    # Prepare dependent variable
    y = Y[y_name].reset_index(drop=True)

    # Bootstrap weights and feature engineering
    feature_engineer = False
    X_train, X_test, y_train, y_test, sw_train, sw_test = bootstrap_weights_feature_eng(X_sub, y, feature_engineer)

    # Run Models
    thresh = 0.7
    save_path = path + 'out/qual_graphs_comparison/'
    class_metrics = run_models(summary_table, save_path, thresh)

    # Save Model Comparisons
    class_metrics.to_csv(path + 'out_final/model_comparison/' + y_name + '_fe=0_auc_th=0.7.csv', index=False)
    

