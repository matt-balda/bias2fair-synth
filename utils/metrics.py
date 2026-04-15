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

def calculate_metrics(y_true, y_pred, y_prob, sensitive_features):
    """
    Calculates all metrics defined in the protocol.
    """
    metrics = {}
    
    # Performance
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    metrics['auc_pr'] = average_precision_score(y_true, y_prob)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Fairness
    metrics['statistical_parity_difference'] = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    metrics['disparate_impact'] = demographic_parity_ratio(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    metrics['equal_opportunity_difference'] = equal_opportunity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
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
