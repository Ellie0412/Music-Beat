# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

FIG_COMP = '/Users/ellie/Documents/Assignments/university-python/music_beat/figures/composite'
os.makedirs(FIG_COMP, exist_ok=True)

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

def plot_one_composite(feat, train_ext, test_ext):
    box_df = pd.concat([
        pd.DataFrame({feat: train_ext[feat], 'Dataset': 'Train'}),
        pd.DataFrame({feat: test_ext[feat],   'Dataset': 'Test'})
    ], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios':[2,1]})
    sns.histplot(data=train_ext, x=feat, kde=True, color='#2E86AB', ax=axes[0], label='Train')
    sns.histplot(data=test_ext,  x=feat, kde=True, color='#F18F01', ax=axes[0], label='Test')
    axes[0].set_title(f'{feat} – Distribution'); axes[0].legend()

    sns.boxplot(data=box_df, y='Dataset', x=feat,
                palette={'Train':'#2E86AB','Test':'#F18F01'}, ax=axes[1])
    axes[1].set_title(f'{feat} – Boxplot'); axes[1].set_ylabel('')
    plt.tight_layout()
    plt.savefig(f'{FIG_COMP}/{feat}.png', dpi=300, bbox_inches='tight')
    plt.close()

def batch_plot_composite(train_ext, test_ext):
    cols = ['Energy_AudioLoudness','Mood_Acoustic','TrackDurationMin',
            'Energy_Acoustic_Ratio','Vocal_Instrument_Balance',
            'MoodRhythm','PerformanceIntensity','RhythmEnergy']
    for c in cols:
        plot_one_composite(c, train_ext, test_ext)