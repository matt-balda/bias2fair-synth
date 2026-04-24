import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, 
    recall_score, precision_score, accuracy_score
)
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equal_opportunity_difference
)

def calculate_metrics(y_true, y_pred, y_prob, sensitive_features, save_path=None):
    """
    Calculates all metrics defined in the protocol.
    """
    if save_path is not None:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame({
            'y_true': np.array(y_true),
            'y_pred': np.array(y_pred),
            'y_prob': np.array(y_prob),
            'sensitive_features': np.array(sensitive_features)
        }).to_csv(save_path, index=False)

    metrics = {}
    
    # Performance
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    metrics['auc_pr'] = average_precision_score(y_true, y_prob)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)  
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Fairness (Signed metrics: Unprivileged - Privileged)
    # 0 = Unprivileged (African-American), 1 = Privileged (Caucasian)
    if isinstance(sensitive_features, pd.Series):
        sensitive_array = sensitive_features.values
    else:
        sensitive_array = np.array(sensitive_features)
        
    priv_mask = (sensitive_array == 1)
    unpriv_mask = (sensitive_array == 0)
    
    y_pred_array = np.array(y_pred)
    y_true_array = np.array(y_true)
    
    pr_priv = y_pred_array[priv_mask].mean() if priv_mask.sum() > 0 else 0.0
    pr_unpriv = y_pred_array[unpriv_mask].mean() if unpriv_mask.sum() > 0 else 0.0
    
    metrics['statistical_parity_difference'] = pr_unpriv - pr_priv
    metrics['disparate_impact'] = pr_unpriv / pr_priv if pr_priv > 0 else np.nan
    
    # TPR for Equal Opportunity on target class 1
    y1_priv_mask = priv_mask & (y_true_array == 1)
    y1_unpriv_mask = unpriv_mask & (y_true_array == 1)
    
    tpr_priv = y_pred_array[y1_priv_mask].mean() if y1_priv_mask.sum() > 0 else 0.0
    tpr_unpriv = y_pred_array[y1_unpriv_mask].mean() if y1_unpriv_mask.sum() > 0 else 0.0
    
    metrics['equal_opportunity_difference'] = tpr_unpriv - tpr_priv
    
    metrics['average_absolute_odds_difference'] = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    
    return metrics

def aggregate_results(results_list):
    """
    Aggregates list of metric dictionaries into mean +/- std.
    """
    df = pd.DataFrame(results_list)
    summary = df.agg(['mean', 'std']).T
    return summary
