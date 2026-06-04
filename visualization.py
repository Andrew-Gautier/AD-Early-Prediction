import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)



def plot_feature_importance(importances, feature_names, top_n=20, title=None, save_path=None,
                            title_fontsize=14, label_fontsize=12, tick_fontsize=10, value_fontsize=8):
    """Publication-quality horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importances : array-like or fitted model
        Either a 1-D array of importance values or a fitted model with
        a `feature_importances_` attribute.
    feature_names : list[str]
        Feature names corresponding to the importance values.
    top_n : int
        Number of top features to display.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    title_fontsize : int
        Font size for the chart title.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels (feature names).
    value_fontsize : int
        Font size for the value annotations on each bar.
    """
    if hasattr(importances, 'feature_importances_'):
        importances = importances.feature_importances_
    fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi = fi.sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(fi)))
    ax.barh(fi['Feature'], fi['Importance'], color=colors, edgecolor='white', linewidth=0.5)
    for i, (val, name) in enumerate(zip(fi['Importance'], fi['Feature'])):
        ax.text(val + fi['Importance'].max() * 0.01, i, f'{val:.4f}', va='center', fontsize=value_fontsize)
    ax.set_xlabel('Importance (gain)', fontsize=label_fontsize)
    ax.set_title(title or f'Top {top_n} Feature Importances', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_mat(y_true, y_pred, class_labels=None, title=None, save_path=None):
    """Publication-quality annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    class_labels : list[str], optional
        Display names for classes (default: ['0', '1']).
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_true))]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
                yticklabels=class_labels, linewidths=0.5, linecolor='gray',
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title or 'Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc(y_true, y_proba, title=None, save_path=None):
    """Publication-quality ROC curve with AUC annotation and diagonal reference.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC curve (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title or 'Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fpr, tpr, auc_val


def plot_pr_curve(y_true, y_proba, title=None, save_path=None):
    """Publication-quality Precision-Recall curve with Average Precision annotation.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#E84855', lw=2.5, label=f'PR curve (AP = {ap:.3f})')
    ax.axhline(baseline, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Random classifier (AP = {baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.15, color='#E84855')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title or 'Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return precision, recall, ap


def plot_shap_summary(model, X_train, feature_names, max_display=20, title=None, save_path=None):
    """SHAP beeswarm summary plot using TreeExplainer (computed on training data).

    Parameters
    ----------
    model : fitted XGBClassifier
        The trained model to explain.
    X_train : np.ndarray
        Training feature matrix (the data SHAP values are computed on).
    feature_names : list[str]
        Feature names corresponding to columns of X_train.
    max_display : int
        Maximum number of features to display (sorted by mean |SHAP|).
    title : str, optional
        Custom plot title (rendered as a suptitle).
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features).
    explainer : shap.TreeExplainer
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # shap.summary_plot manages its own figure; use show=False so we can save
    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return shap_values, explainer
