import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import skew

# 1. 读入数据
train = pd.read_csv('/Users/ellie/Documents/Assets/csv/playground-series-s5e9/train.csv')
test = pd.read_csv('/Users/ellie/Documents/Assets/csv/playground-series-s5e9/test.csv')

# 2. 去掉 id 列
# 2. 提前保存 test 的 id，然后再去掉 id 列
test_id = test[['id']].copy()
train = train.drop(columns=['id'])
test  = test.drop(columns=['id'])

print('train:', train.shape)   # (行数, 列数)
print('test :', test.shape)
# 3. 统一变量类型（type）
num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent',
            'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood',
            'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute']

# test 没有 BeatsPerMinute，需要单独处理
test_num_cols = [c for c in num_cols if c in test.columns]

# 强制转换为数值型，无法转换的变成 NaN
train[num_cols] = train[num_cols].apply(pd.to_numeric, errors='coerce')
test[test_num_cols]  = test[test_num_cols].apply(pd.to_numeric, errors='coerce')

# 4. 查看 info / describe
print('===== train.info =====')
print(train.info())
print('\n===== test.info =====')
print(test.info())

print('\n===== train.describe =====')
print(train.describe())
print('\n===== test.describe =====')
print(test.describe())

# 5. 缺失值、重复值
print('\n===== 缺失值 =====')
print('train:\n', train.isna().sum())
print('test:\n', test.isna().sum())



print('\n===== 重复行 =====')
print('train 重复行数:', train.duplicated().sum())
print('test  重复行数:', test.duplicated().sum())



# 5. 逐列 IQR 异常值检测函数
def outliers_per_column(df, columns):
    res = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        res[col] = cnt
    return pd.Series(res, name='outlier_count')

# 2. 对两个数据集分别统计
train_cols = train.select_dtypes(include=np.number).columns
test_cols  = test.select_dtypes(include=np.number).columns

train_out = outliers_per_column(train, train_cols)
test_out  = outliers_per_column(test,  test_cols)

print('===== train 各列异常值数量 =====')
print(train_out)
print('\n===== test 各列异常值数量 =====')
print(test_out)













# # ===== 训练集 BeatsPerMinute 分布图 & 箱形图 =====
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# sns.histplot(train['BeatsPerMinute'], kde=True, bins=30)
# plt.title('BeatsPerMinute Distribution (train)')
# plt.xlabel('BeatsPerMinute')
# plt.ylabel('Count')

# plt.subplot(1, 2, 2)
# sns.boxplot(y=train['BeatsPerMinute'])
# plt.title('BeatsPerMinute Boxplot (train)')
# plt.ylabel('BeatsPerMinute')

# plt.tight_layout()
# plt.show()





# ---------- 3. 单特征大图 ----------

# 在轮播前加一行：统一保存路径
import os
os.makedirs('composite_plots', exist_ok=True)   # 保存目录


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每个组合特征单独一张 1×2 大图：
左：分布直方图（含 KDE）
右：横向箱形图（Merged vs Test）
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. 组合特征构造 ----------
def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Energy_AudioLoudness"]      = df["Energy"] * df["AudioLoudness"]
    df["Mood_Acoustic"]             = df["MoodScore"] * df["AcousticQuality"]
    df["TrackDurationMin"]          = df["TrackDurationMs"] / 60_000
    df["Energy_Acoustic_Ratio"]     = df["Energy"] / (df["AcousticQuality"] + 1e-5)
    df["Vocal_Instrument_Balance"]  = df["VocalContent"] / (df["InstrumentalScore"] + 1e-5)
    df["MoodRhythm"]                = df["MoodScore"] * df["RhythmScore"]
    df["PerformanceIntensity"]      = df["LivePerformanceLikelihood"] * df["AudioLoudness"]
    df["RhythmEnergy"]              = df["RhythmScore"] * df["Energy"]
    return df

# ---------- 2. 数据准备 ----------
train_ext = add_composite(train)
test_ext  = add_composite(test)

# 为箱形图标记来源（临时 DataFrame，不污染原表）
def make_box_df(feat):
    return pd.concat([
        pd.DataFrame({feat: train_ext[feat], 'Dataset': 'Train'}),
        pd.DataFrame({feat: test_ext[feat],   'Dataset': 'Test'})
    ], ignore_index=True)

    
# def plot_one_composite(feat: str):
#     box_df = make_box_df(feat)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2, 1]})

#     # 左：直方图 + KDE
#     sns.histplot(data=train_ext, x=feat, kde=True, color='#2E86AB', ax=axes[0], label='Train')
#     sns.histplot(data=test_ext,  x=feat, kde=True, color='#F18F01', ax=axes[0], label='Test')
#     axes[0].set_title(f'{feat} – Distribution')
#     axes[0].legend()

#     # 右：横向箱形图
#     sns.boxplot(data=box_df, y='Dataset', x=feat, palette={'Train': '#2E86AB', 'Test': '#F18F01'}, ax=axes[1])
#     axes[1].set_title(f'{feat} – Boxplot')
#     axes[1].set_ylabel('')

#     plt.tight_layout()
#     # plt.show()          # 先注释掉
#     plt.savefig(f'/Users/ellie/Documents/Assignments/university-python/music_beat/composite_plots/{feat}.png', dpi=300, bbox_inches='tight')
#     plt.close()           # 务必关闭，释放内存

# # ---------- 4. 轮播每个组合特征 ----------
# composite_cols = [
#     'Energy_AudioLoudness', 'Mood_Acoustic', 'TrackDurationMin',
#     'Energy_Acoustic_Ratio', 'Vocal_Instrument_Balance',
#     'MoodRhythm', 'PerformanceIntensity', 'RhythmEnergy'
# ]

# for col in composite_cols:
#     plot_one_composite(col)





























import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. 组合特征构造 ----------
def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Energy_AudioLoudness"]      = df["Energy"] * df["AudioLoudness"]
    df["Mood_Acoustic"]             = df["MoodScore"] * df["AcousticQuality"]
    df["TrackDurationMin"]          = df["TrackDurationMs"] / 60_000
    df["Energy_Acoustic_Ratio"]     = df["Energy"] / (df["AcousticQuality"] + 1e-5)
    df["Vocal_Instrument_Balance"]  = df["VocalContent"] / (df["InstrumentalScore"] + 1e-5)
    df["MoodRhythm"]                = df["MoodScore"] * df["RhythmScore"]
    df["PerformanceIntensity"]      = df["LivePerformanceLikelihood"] * df["AudioLoudness"]
    df["RhythmEnergy"]              = df["RhythmScore"] * df["Energy"]
    return df

# ---------- 2. 数据准备 ----------
# ---------- 1. 扩展特征 ----------
train_ext = add_composite(train)  
test_ext  = add_composite(test)  



df_merge  = pd.concat([train_ext, test_ext], axis=0, ignore_index=True)

# ---------- 3. 绘图函数 ----------
def plot_lower_corr(df, title, figsize=(10, 8)):
    corr = df.select_dtypes(include='number').corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", linewidths=.5,
                cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .8})
    plt.title(title, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()
    # plt.close()

# ---------- 4. 一键出图 ----------
plot_lower_corr(df_merge, "Merged Data – Original + Composite Features (Lower)")
plot_lower_corr(test_ext, "Test Data – Original + Composite Features (Lower)")










#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集异常值扫描（IQR 1.5×）
从零开始：加组合特征 → 扫描 → 打印
"""

import pandas as pd
import numpy as np

# ---------- 1. 组合特征构造 ----------
def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Energy_AudioLoudness"]      = df["Energy"] * df["AudioLoudness"]
    df["Mood_Acoustic"]             = df["MoodScore"] * df["AcousticQuality"]
    df["TrackDurationMin"]          = df["TrackDurationMs"] / 60_000
    df["Energy_Acoustic_Ratio"]     = df["Energy"] / (df["AcousticQuality"] + 1e-5)
    df["Vocal_Instrument_Balance"]  = df["VocalContent"] / (df["InstrumentalScore"] + 1e-5)
    df["MoodRhythm"]                = df["MoodScore"] * df["RhythmScore"]
    df["PerformanceIntensity"]      = df["LivePerformanceLikelihood"] * df["AudioLoudness"]
    df["RhythmEnergy"]              = df["RhythmScore"] * df["Energy"]
    return df

# ---------- 2. 现场生成测试集 ----------
test_ext = add_composite(test)          # 只要原始 test 存在即可

# ---------- 3. 异常值扫描函数 ----------
def checking_outlier(list_feature, df, dataset_name):
    print(f"\n>>> Outlier Check for {dataset_name} (IQR 1.5×)")
    print("-" * 50)
    print(f"{'Feature':<35} {'Count':>8} {'Ratio':>8}")
    print("-" * 50)
    total = len(df)
    for feat in list_feature:
        if feat not in df.columns:
            print(f"{feat:<35} {'N/A':>8} {'N/A':>8}")
            continue
        s = df[feat]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        out_cnt = ((s < lower) | (s > upper)).sum()
        print(f"{feat:<35} {out_cnt:>8} {out_cnt/total:>7.2%}")
    print("-" * 50)

# ---------- 4. 一键扫描 ----------
num_cols = test_ext.select_dtypes(include='number').columns.tolist()
checking_outlier(list_feature=num_cols, df=test_ext, dataset_name="Test data")






def plot_train_test_feature(col: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    对比 train/test 单个特征的分布
    左：叠加直方图 + KDE
    右：横向箱线图
    """
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
    plt.show()
    # plt.close()

# 例：逐个查看所有数值特征
num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent',
            'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood',
            'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute']

for feat in num_cols:
    if feat in train.columns and feat in test.columns:
        plot_train_test_feature(feat, train, test)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr_joint(train: pd.DataFrame,
                    test:  pd.DataFrame,
                    figsize=(16, 6),
                    save_path=None):
    """
    一张大图：左右各一个热力图，分别展示 train / test 数值特征的 Pearson 相关系数
    """
    # 1. 各自数值列
    train_num = train.select_dtypes(include='number')
    test_num  = test.select_dtypes(include='number')

    # 2. 计算相关系数
    corr_train = train_num.corr(method='pearson')
    corr_test  = test_num.corr(method='pearson')

    # 3. 画图
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(corr_train, cmap='coolwarm', center=0,
                annot=False, linewidths=0.5, square=True, ax=axes[0])
    axes[0].set_title('Train Correlation Matrix')

    sns.heatmap(corr_test, cmap='coolwarm', center=0,
                annot=False, linewidths=0.5, square=True, ax=axes[1])
    axes[1].set_title('Test Correlation Matrix')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    # plt.close()

# 直接调用
plot_corr_joint(train, test)

































# ================= 1. 额外 import =================
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor


# ================= 2. 重新切一份 validation =================
# 这里直接用你已经造好特征的 train_ext
X = train_ext.drop(columns=['BeatsPerMinute'])
y = train_ext['BeatsPerMinute']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= 3. 把 SHAP 画图函数搬过来 =================
def shap_plot(model, X_test, list_feature, type=None):
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    X_test_sample = pd.DataFrame(X_test, columns=list_feature)
    explainer = shap.Explainer(model.predict, X_test_sample)
    shap_values = explainer(X_test_sample)
    if type == "bar":
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": X_test_sample.columns,
                                "importance": shap_importance})
        shap_df = shap_df.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=shap_df["importance"], y=shap_df["feature"],
                    palette="viridis", order=shap_df["feature"])
        plt.xlabel("mean(|SHAP value|)")
        plt.title("SHAP Feature Importance", fontsize=14, weight="bold", pad=20)
        plt.tight_layout()
        plt.show()
    else:
        shap.summary_plot(shap_values, X_test_sample)

# ================= 4. 把评估函数搬过来 =================
def evaluate_model(model, X_train, X_val, y_train, y_val, show_shap_plot=True):
    RESET = "\033[0m"
    BLUE = "\033[94m"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Model: {model.__class__.__name__}{RESET}")
    print(f"Root Mean Squared Error (RMSE): {BLUE}{rmse:.4f}{RESET}")
    print("-" * 80)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    # ----- Plot 1: Predicted vs. Actual -----
    axs[0].scatter(y_val, y_pred, alpha=0.4, color="royalblue")
    axs[0].plot([y_val.min(), y_val.max()],
                [y_val.min(), y_val.max()],
                "r--", lw=2, label="Perfect Prediction (y=x)")
    axs[0].set_xlabel("Actual Values (BeatsPerMinute)")
    axs[0].set_ylabel("Predicted Values (BeatsPerMinute)")
    axs[0].set_title("Predicted vs. Actual (Validation Set)",
                     fontsize=14, weight="bold", pad=20)
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)

    # ----- Plot 2: Residual Plot -----
    residuals = y_val - y_pred
    axs[1].scatter(y_val, residuals, alpha=0.5)
    axs[1].axhline(0, color="red", linestyle="--", lw=2)
    axs[1].set_xlabel("Actual Values (BeatsPerMinute)")
    axs[1].set_ylabel("Prediction Error (Residuals)")
    axs[1].set_title("Residual Plot", fontsize=14, weight="bold", pad=20)
    axs[1].grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    if show_shap_plot:
        shap_plot(model=model, X_test=X_val, list_feature=X_val.columns)

# ================= 5. 训练 + 评估 =================
# 用你之前 Optuna 调好的超参
param_cb = {
    "iterations": 601,
    "learning_rate": 0.010916330886941803,
    "depth": 5,
    "l2_leaf_reg": 90.94596820625567,
    "random_strength": 1.8922481051459825,
    "border_count": 218,
    "leaf_estimation_iterations": 6,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.7711311287541387,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0
}

model_cb = CatBoostRegressor(**param_cb)

# 一键出图
evaluate_model(model_cb, X_train, X_val, y_train, y_val, show_shap_plot=True)