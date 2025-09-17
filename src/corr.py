# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

FIG_CORR = '/Users/ellie/Documents/Assignments/university-python/music_beat/figures/corr'
os.makedirs(FIG_CORR, exist_ok=True)

def plot_lower_corr(df, title_suffix):
    corr = df.select_dtypes(include='number').corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=.5,
                cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .8})
    plt.title(f"Lower Triangular Corr – {title_suffix}")
    plt.tight_layout()
    plt.savefig(f'{FIG_CORR}/corr_{title_suffix.replace(" ","_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_joint_corr(train, test):
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    sns.heatmap(train.corr(), cmap='coolwarm', center=0, linewidths=.5,
                square=True, ax=axes[0], cbar_kws={"shrink": .8})
    axes[0].set_title('Train Correlation')
    sns.heatmap(test.corr(),  cmap='coolwarm', center=0, linewidths=.5,
                square=True, ax=axes[1], cbar_kws={"shrink": .8})
    axes[1].set_title('Test Correlation')
    plt.tight_layout()
    plt.savefig(f'{FIG_CORR}/train_vs_test_corr.png', dpi=300, bbox_inches='tight')
    plt.close()