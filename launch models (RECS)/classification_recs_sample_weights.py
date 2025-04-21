
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Ensemble Methods
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.ensemble import (
    BaggingClassifier, 
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import KFold, train_test_split

# SVC
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve, 
    average_precision_score, precision_score, recall_score,
    cohen_kappa_score, f1_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# --- Resampling Function ---
def resample(X_sub_train, y_sub_train, os_type):
    """
    Resample the dataset to balance class distribution.
    os_type: 'smote', 'over', 'under', or 'none'
    """
    if os_type == 'smote':
        oversample = SMOTE()
        X_sub_train, y_sub_train = oversample.fit_resample(X_sub_train, y_sub_train)
    elif os_type == 'over':
        over = RandomOverSampler(sampling_strategy=0.5)
        X_sub_train, y_sub_train = over.fit_resample(X_sub_train, y_sub_train)
    elif os_type == 'under':
        under = RandomUnderSampler(sampling_strategy=0.5)
        X_sub_train, y_sub_train = under.fit_resample(X_sub_train, y_sub_train)
    # Return resampled data
    return X_sub_train, y_sub_train


# --- ROC Curve and Precision-Recall Curve Plotting ---
def ROC_curve(y_test, list_models_prob):
    """
    Plot ROC and Precision-Recall curves for multiple models.
    """
    score_auc = {}
    average_precision = {}
    fs = 10  # Font size for plot labels

    # Plot ROC Curve
    for key, value in list_models_prob.items():
        y_prob = value[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=key)

    plt.title('ROC Curve', fontsize=fs)
    plt.xlabel('False Positive Rate', fontsize=fs)
    plt.ylabel('True Positive Rate', fontsize=fs)
    plt.legend(loc='best')
    plt.show()

    # Plot Precision-Recall Curve
    for key, value in list_models_prob.items():
        y_prob = value[:, 1]
        prec, recall, _ = precision_recall_curve(y_test, y_prob)
        PrecisionRecallDisplay(precision=prec, recall=recall).plot()

        plt.title(f'Precision-Recall Curve - {key}', fontsize=fs)
        plt.xlabel('Recall', fontsize=fs)
        plt.ylabel('Precision', fontsize=fs)
        plt.legend(loc='best')
        plt.show()

        # Store scores
        score_auc[key] = roc_auc_score(y_test, y_prob)
        average_precision[key] = average_precision_score(y_test, y_prob)

    return score_auc, average_precision


# --- Logistic Regression Fit ---
def logistic_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table, thresh=0.5):
    """
    Fit Logistic Regression and evaluate its performance.
    """
    logreg = LogisticRegression(class_weight="balanced")
    logreg.fit(X_train, y_train, sample_weight=weights_train)

    # Predict and calculate probabilities
    y_pred = logreg.predict(X_test)
    proba = logreg.predict_proba(X_test)

    # Apply threshold if necessary
    if thresh != 0.5:
        y_pred = (proba[:, 1] >= thresh).astype(int)

    # Evaluate model performance
    score = accuracy_score(y_test, y_pred, sample_weight=weights_test)
    report = classification_report(y_test, y_pred, sample_weight=weights_test)
    print("Logistic Regression Performance:\n", report)
    print(confusion_matrix(y_test, y_pred, sample_weight=weights_test))

    # Add model performance to summary table
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['LogisticReg'], 'score': [score]})], ignore_index=True)

    return logreg, summary_table, proba


# --- Ensemble Fit Function ---
def ensemble_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table, thresh=0.5):
    """
    Train and evaluate various ensemble models.
    Returns models and predictions with updated summary table.
    """
    rand_seed = 5
    test_prop = 0.2

    # Split train set into training and validation sets
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X_train, y_train, weights_train, test_size=test_prop, random_state=1
    )

    # --- Bagging Classifier ---
    n_est = 150
    max_sam = 60
    max_feat = 10

    bag_tree = BaggingClassifier(random_state=rand_seed, n_estimators=n_est, max_samples=max_sam, max_features=max_feat)
    bag_tree.fit(X_train, y_train, sample_weight=weights_train)

    y_pred1 = bag_tree.predict(X_test)
    proba1 = bag_tree.predict_proba(X_test)

    if thresh != 0.5:
        y_pred1 = (proba1[:, 1] >= thresh).astype(int)

    # Evaluate model and add to summary table
    bag_test_score = bag_tree.score(X_test, y_test, sample_weight=weights_test)
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['Bagging'], 'score': [bag_test_score]})], ignore_index=True)

    # --- Random Forest Classifier ---
    n_est2 = 25
    max_sam2 = 30
    max_feat2 = "sqrt"

    rf_tree = RandomForestClassifier(random_state=rand_seed, max_depth=15, n_estimators=n_est2, max_samples=max_sam2, max_features=max_feat2)
    rf_tree.fit(X_train, y_train, sample_weight=weights_train)

    y_pred2 = rf_tree.predict(X_test)
    proba2 = rf_tree.predict_proba(X_test)

    if thresh != 0.5:
        y_pred2 = (proba2[:, 1] >= thresh).astype(int)

    # Evaluate model and add to summary table
    rf_test_score = rf_tree.score(X_test, y_test, sample_weight=weights_test)
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['RandomForest'], 'score': [rf_test_score]})], ignore_index=True)

    # --- Gradient Boosting Classifier ---
    gb_tree = GradientBoostingClassifier(random_state=rand_seed)
    gb_tree.fit(X_train, y_train, sample_weight=weights_train)

    y_pred3 = gb_tree.predict(X_test)
    proba3 = gb_tree.predict_proba(X_test)

    if thresh != 0.5:
        y_pred3 = (proba3[:, 1] >= thresh).astype(int)

    gb_test_score = gb_tree.score(X_test, y_test, sample_weight=weights_test)
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['GBoosting'], 'score': [gb_test_score]})], ignore_index=True)

    # --- AdaBoost Classifier ---
    ab_tree = AdaBoostClassifier(random_state=rand_seed, n_estimators=50)
    ab_tree.fit(X_train, y_train, sample_weight=weights_train)

    y_pred4 = ab_tree.predict(X_test)
    proba4 = ab_tree.predict_proba(X_test)

    if thresh != 0.5:
        y_pred4 = (proba4[:, 1] >= thresh).astype(int)

    ab_test_score = ab_tree.score(X_test, y_test, sample_weight=weights_test)
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['AdaBoosting'], 'score': [ab_test_score]})], ignore_index=True)

    return bag_tree, rf_tree, gb_tree, ab_tree, summary_table, proba1, proba2, proba3, proba4


# --- SVC Fit ---
def SVC_fit(X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, thresh=0.5):
    """
    Fit Support Vector Classifier with Grid Search.
    """
    param_grid = {'C': [5, 16, 18, 30], 'kernel': ['rbf', 'sigmoid'], 'gamma': [0.0001, 0.001, 0.01]}
    grid = GridSearchCV(SVC(probability=True, class_weight="balanced"), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    proba = grid.predict_proba(X_test)

    if thresh != 0.5:
        y_pred = (proba[:, 1] >= thresh).astype(int)

    score = accuracy_score(y_pred, y_test, sample_weight=sw_test)
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['SVC'], 'score': [score]})], ignore_index=True)

    return grid, summary_table, proba


# --- KNN Fit ---
def KNN_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table, splits, thresh=0.5):
    # Ensure input data is properly formatted
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values
    if not isinstance(X_test, np.ndarray):
        X_test = X_test.values
    
    # Make arrays C-contiguous
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    mses_knn = []
    for n in range(0, 20):
        knn_cv = KNeighborsClassifier(n_neighbors=n+1)
        kfold = KFold(n_splits=splits)

        # Cross-validation
        scores = cross_val_score(knn_cv, X_train, y_train, cv=kfold)
        mses_knn.append(scores.mean())

    # Get the best K value
    K = mses_knn.index(max(mses_knn)) + 1
    print(f"Optimal K: {K}")

    # Fit KNN with best K
    knn_cv = KNeighborsClassifier(n_neighbors=K)
    knn_cv.fit(X_train, y_train)

    # Predict using KNN
    y_pred = knn_cv.predict(X_test)

    # Get accuracy and classification report
    score = accuracy_score(y_pred, y_test, sample_weight=weights_test)
    report = classification_report(y_test, y_pred, sample_weight=weights_test)

    print(confusion_matrix(y_test, y_pred, sample_weight=weights_test))
    print(report)

    # Update summary table
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['KNN'], 'score': [score]})], ignore_index=True)

    return knn_cv, summary_table, knn_cv.predict_proba(X_test)


# --- ANN Fit ---
def ANN_fit(X_train, y_train, X_test, y_test, weights_train, weights_test, summary_table, thresh=0.5):
    """
    Fit Artificial Neural Network (MLP) with Grid Search.
    """
    mlp_gs = MLPClassifier(max_iter=100)
    parameter_space = {'hidden_layer_sizes': [(10, 30, 10), (20,), (64, 64, 64)], 'activation': ['relu', 'sigmoid']}

    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    if thresh != 0.5:
        y_pred = (proba[:, 1] >= thresh).astype(int)

    score = accuracy_score(y_pred, y_test, sample_weight=weights_test)
    summary_table = pd.concat([summary_table, pd.DataFrame({'Model': ['ANN'], 'score': [score]})], ignore_index=True)

    return clf, summary_table, proba
