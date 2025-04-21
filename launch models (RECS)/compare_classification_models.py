import os
import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, 
    matthews_corrcoef, accuracy_score, precision_score, recall_score, 
    cohen_kappa_score, f1_score, auc
)
import seaborn as sns


# --- Confusion Matrix Values ---
def confusion_matrix_values(y_true, y_score, decision_thresh):
    """
    Obtain confusion matrix values (TP, FP, TN, FN) based on decision threshold.
    """
    y_pred = (np.array(y_score) > decision_thresh)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    
    true_neg, false_pos, false_neg, true_pos = cm.ravel()
    return true_neg, false_pos, false_neg, true_pos


# --- Calculate TPR, FPR, and Precision ---
def calculate_tpr_fpr_prec(y_true, y_score, decision_thresh):
    """
    Calculate True Positive Rate (TPR), False Positive Rate (FPR), and Precision
    for a given decision threshold.
    """
    true_neg, false_pos, false_neg, true_pos = confusion_matrix_values(y_true, y_score, decision_thresh)
    tpr_recall = float(true_pos) / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    fpr = float(false_pos) / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0
    precision = float(true_pos) / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    return tpr_recall, fpr, precision


# --- PlotAll Class ---
class PlotAll(object):
    def __init__(self, title, savetitle, y_true, y_score, path_save, thresh):
        """
        Initialize PlotAll class with required inputs and call plot functions.
        """
        self.y_true = y_true
        self.y_score = y_score
        self.thresh = thresh
        
        # Style Parameters
        self.roc_color = 'crimson'
        self.pr_color = 'royalblue'
        self.main_linestyle = 'solid'
        self.neutral_color = 'k'
        self.neutral_linestyle = 'dashed'
        self.lw = 1
        self.y_pred = (np.array(self.y_score) > thresh)
        
        # Create Subplots
        fig, self.ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5.5))
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_confusion_matrix()
        
        # Add Title and Save Plot
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0.10, 1, 0.90])
        plt.savefig(os.path.join(path_save, savetitle + '_th=0.7.png'), transparent=True)

    # --- Plot ROC Curve ---
    def plot_roc_curve(self):
        """
        Plot the ROC curve and calculate AUC.
        """
        fs = 10
        fpr, tpr_recall, _ = sklearn.metrics.roc_curve(self.y_true, self.y_score, pos_label=1)
        
        # Plot ROC Curve
        self.ax[0].plot(fpr, tpr_recall, color=self.roc_color, lw=self.lw, linestyle=self.main_linestyle)
        self.ax[0].fill_between(fpr, tpr_recall, step='post', alpha=0.2, color=self.roc_color)
        self.ax[0].plot([0, 1], [0, 1], color=self.neutral_color, lw=self.lw, linestyle=self.neutral_linestyle)
        
        # Set Plot Limits and Labels
        self.ax[0].set_xlim([0.0, 1.0])
        self.ax[0].set_ylim([0.0, 1.05])
        self.ax[0].set_xlabel('False Positive Rate', fontsize=fs)
        self.ax[0].set_ylabel('True Positive Rate (Recall)', fontsize=fs)
        
        # Calculate AUC for ROC
        self.auc_roc = auc(fpr, tpr_recall)
        self.ax[0].set_title(f'AUROC = {self.auc_roc:.2f}')
        
        # Plot Decision Thresholds for Clarity
        for d in [0.1, 0.5, 0.9]:
            tpr_recall, fpr, _ = calculate_tpr_fpr_prec(self.y_true, self.y_score, d)
            self.ax[0].plot(fpr, tpr_recall, 'o', color=self.roc_color)
            self.ax[0].annotate(f'd={d}', (fpr, tpr_recall))

    # --- Plot Precision-Recall Curve ---
    def plot_precision_recall_curve(self):
        """
        Plot Precision-Recall curve and calculate AUC.
        """
        fs = 10
        precision, tpr_recall, _ = sklearn.metrics.precision_recall_curve(self.y_true, self.y_score)
        
        # Plot Precision-Recall Curve
        self.ax[1].step(tpr_recall, precision, color=self.pr_color, alpha=0.2, where='post', linewidth=self.lw, linestyle=self.main_linestyle)
        self.ax[1].fill_between(tpr_recall, precision, step='post', alpha=0.2, color=self.pr_color)
        
        # Set Plot Limits and Labels
        self.ax[1].set_xlabel('True Positive Rate (Recall)', fontsize=fs)
        self.ax[1].set_ylabel('Precision', fontsize=fs)
        self.ax[1].set_ylim([0.0, 1.05])
        self.ax[1].set_xlim([0.0, 1.0])
        
        # Calculate AUC for Precision-Recall
        self.auc_pr = auc(tpr_recall, precision)
        self.ax[1].set_title(f'AUC = {self.auc_pr:.2f}')
        
        # Plot Decision Thresholds for Clarity
        for d in [0.1, 0.5, 0.9]:
            tpr_recall, _, precision = calculate_tpr_fpr_prec(self.y_true, self.y_score, d)
            self.ax[1].plot(tpr_recall, precision, 'o', color=self.pr_color)
            text = self.ax[1].annotate(f'd={d}', (tpr_recall, precision))
            text.set_rotation(45)

    # --- Plot Confusion Matrix ---
    def plot_confusion_matrix(self):
        """
        Plot Confusion Matrix with labels and percentages.
        """
        cm = sklearn.metrics.confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
        group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
        
        # Format Confusion Matrix Labels
        group_counts = [f"{value:0.0f}" for value in cm.flatten()]
        group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        # Plot Confusion Matrix Heatmap
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=self.ax[2], vmin=0, vmax=np.max(cm))
        self.ax[2].set_title('Confusion Matrix')

    # --- Get Metrics ---
    def get_metrics(self):
        """
        Return classification metrics along with AUC for ROC and Precision-Recall.
        """
        return (
            matthews_corrcoef(self.y_true, self.y_pred),
            accuracy_score(self.y_true, self.y_pred),
            f1_score(self.y_true, self.y_pred),
            precision_score(self.y_true, self.y_pred),
            recall_score(self.y_true, self.y_pred),
            cohen_kappa_score(self.y_true, self.y_pred),
            self.auc_roc,  # Return AUC for ROC
            self.auc_pr    # Return AUC for Precision-Recall
        )


