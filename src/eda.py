# -*- coding: utf-8 -*-
import pandas as pd
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
import os

FIG_EDA = '/Users/ellie/Documents/Assignments/university-python/music_beat/figures/eda'
os.makedirs(FIG_EDA, exist_ok=True)

def outliers_per_column(df, columns):
    res = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        res[col] = cnt
    return pd.Series(res, name='outlier_count')

def quick_overview(train, test):
    print('>>> 形状与缺失')
    print('train:', train.shape, 'test:', test.shape)
    print('\n>>> 缺失值')
    print('train:\n', train.isna().sum())
    print('test:\n', test.isna().sum())
    print('\n>>> 重复行')
    print('train 重复:', train.duplicated().sum())
    print('test  重复:', test.duplicated().sum())

    train_out = outliers_per_column(train, train.select_dtypes(include='number').columns)
    test_out  = outliers_per_column(test,  test.select_dtypes(include='number').columns)
    print('\n>>> 异常值(IQR)')
    print('train:\n', train_out)
    print('test:\n', test_out)

def bpm_dist(train):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.histplot(train['BeatsPerMinute'], kde=True, bins=30)
    plt.title('BeatsPerMinute Distribution')
    plt.subplot(1,2,2)
    sns.boxplot(y=train['BeatsPerMinute'])
    plt.title('BeatsPerMinute Boxplot')
    plt.tight_layout()
    plt.savefig(f'{FIG_EDA}/BeatsPerMinute.png', dpi=300, bbox_inches='tight')
    plt.close()




    