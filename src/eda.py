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



# 单特征 train/test 对比图
def plot_train_test_feature(col: str,
                            train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            save_dir: str = FIG_EDA):
    if col not in train_df.columns or col not in test_df.columns:
        print(f'{col} 不存在于某个数据集，跳过')
        return

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 1. 叠加直方图 + KDE
    sns.histplot(train_df[col], bins=40, kde=True, color='#F6BEC8',
                 alpha=0.5, line_kws={'linewidth': 2}, ax=axes[0], label='train')
    sns.histplot(test_df[col], bins=40, kde=True, color='#78974B',
                 alpha=0.5, line_kws={'linewidth': 2}, ax=axes[0], label='test')
    axes[0].set_title(f'{col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # 2. 横向箱线图
    box_data = pd.concat([
        train_df[[col]].assign(dataset='train'),
        test_df[[col]].assign(dataset='test')
    ])
    sns.boxplot(data=box_data, x=col, y='dataset', ax=axes[1],
                palette={'train': '#F6BEC8', 'test': '#78974B'})
    axes[1].set_title(f'{col}')
    axes[1].set_xlabel(col)
    axes[1].set_ylabel('')

    plt.tight_layout()
    out_path = os.path.join(save_dir, f'train_test_{col}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()          # 必须关闭，防止内存泄漏
    print(f'[eda] 已保存 {out_path}')

#批量调用
NUM_FEATURES = [
    'RhythmScore', 'AudioLoudness', 'VocalContent',
    'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood',
    'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute'
]

def batch_plot_train_test(train_df, test_df, save_dir: str = FIG_EDA):
    """对 NUM_FEATURES 中同时存在于 train/test 的列出图"""
    os.makedirs(save_dir, exist_ok=True)
    for feat in NUM_FEATURES:
        if feat in train_df.columns and feat in test_df.columns:
            plot_train_test_feature(feat, train_df, test_df, save_dir)

    
