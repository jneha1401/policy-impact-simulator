import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def calculate_comprehensive_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'sMAPE': smape}

def generate_metric_report(model_results_dict):
    records = []
    for scenario, data in model_results_dict.items():
        metrics = calculate_comprehensive_metrics(data['true'], data['pred'])
        metrics['Scenario'] = scenario
        records.append(metrics)
    return pd.DataFrame(records).set_index('Scenario')
