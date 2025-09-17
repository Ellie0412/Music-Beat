# -*- coding: utf-8 -*-
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import pandas as pd
import os

FIG_MODEL = '/Users/ellie/Documents/Assignments/university-python/music_beat/figures/model'
os.makedirs(FIG_MODEL, exist_ok=True)

def shap_save(model, X_val, feat_list, kind='bar'):
    if hasattr(X_val, "toarray"):
        X_val = X_val.toarray()
    X_sample = pd.DataFrame(X_val, columns=feat_list)
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    if kind == 'bar':
        shap_importance = np.abs(shap_values.values).mean(0)
        df_shap = pd.DataFrame({'feature': feat_list, 'importance': shap_importance})\
                    .sort_values('importance', ascending=False).head(20)
        plt.figure(figsize=(12,6))
        sns.barplot(x='importance', y='feature', data=df_shap, palette='viridis')
        plt.title('SHAP Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(f'{FIG_MODEL}/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(f'{FIG_MODEL}/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_eval_save(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    params = dict(iterations=601, learning_rate=0.0109, depth=5, l2_leaf_reg=90.9,
                  random_strength=1.89, border_count=218,
                  leaf_estimation_iterations=6, bootstrap_type='Bernoulli', subsample=0.77,
                  loss_function='RMSE', eval_metric='RMSE', random_seed=42, verbose=0)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'CatBoost RMSE: {rmse:.4f}')

    # 评估图
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    axes[0].scatter(y_val, y_pred, alpha=0.4, color='royalblue')
    axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted'); axes[0].set_title('Predicted vs Actual')
    axes[0].grid(True, alpha=0.2)
    residuals = y_val - y_pred
    axes[1].scatter(y_val, residuals, alpha=0.5)
    axes[1].axhline(0, color='red', ls='--', lw=2)
    axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Residuals'); axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIG_MODEL}/eval_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    shap_save(model, X_val, X_val.columns, kind='bar')
    return model