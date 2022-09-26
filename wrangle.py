from tracemalloc import start
import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt
import seaborn as sns

# working with dates
from datetime import datetime

def acquire():
    return pd.read_csv('state.csv')

def prepare(state):
    state = state[state.State == 'Illinois']
    state = state.dropna()
    state.columns = ['date', 'avg_temp', 'temp_uncertainty', 'state', 'country']
    state.date = pd.to_datetime(state.date, format='%Y-%m-%d')
    state = state.set_index(state.date).sort_index()
    state.avg_temp = (state.avg_temp * (9/5))+32
    state.temp_uncertainty = (state.temp_uncertainty * (9/5))+32
    state = state.drop(columns=['date', 'state', 'country'])
    state = state.asfreq('M', method='ffill')
    state['month'] = state.index.month_name()
    state['year'] = state.index.year
    
    return state

def split_data(df):
    train_size = 0.80
    n = df.shape[0]
    test_start = round(train_size*n)
    train = df[:test_start]
    test = df[test_start:]

    train_size = 0.75
    n = train.shape[0]
    val_start = round(train_size*n)
    val = train[val_start:]
    train = train[:val_start]
    
    return train, val, test
    