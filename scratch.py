import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1])
y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1, 0.6, 0.75])
sensitive = np.array([1, 1, 0, 0, 1, 0, 0, 1])

res = calculate_metrics(y_true, y_pred, y_prob, sensitive)
print(res)
