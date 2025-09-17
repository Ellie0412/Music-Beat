# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

RAW_TRAIN = '/Users/ellie/Documents/Assets/csv/playground-series-s5e9/train.csv'
RAW_TEST  = '/Users/ellie/Documents/Assets/csv/playground-series-s5e9/test.csv'

def load():
    train = pd.read_csv(RAW_TRAIN)
    test  = pd.read_csv(RAW_TEST)
    test_id = test[['id']].copy()
    train = train.drop(columns=['id'])
    test  = test.drop(columns=['id'])
    return train, test, test_id

def coerce_numeric(train, test):
    num_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent',
                'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood',
                'MoodScore', 'TrackDurationMs', 'Energy', 'BeatsPerMinute']
    test_num_cols = [c for c in num_cols if c in test.columns]
    train[num_cols] = train[num_cols].apply(pd.to_numeric, errors='coerce')
    test[test_num_cols] = test[test_num_cols].apply(pd.to_numeric, errors='coerce')
    return train, test