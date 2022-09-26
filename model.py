import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt
import seaborn as sns

# working with dates
from datetime import datetime

# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# holt's linear trend model. 
from statsmodels.tsa.api import Holt




def evaluate(target_var, validate, yhat):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(mean_squared_error(validate[target_var], yhat[target_var], squared=False), 0)
    return rmse


def plot_and_eval(target_var, train, validate, yhat, title):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat[target_var], label='Prediction', linewidth=1)
    plt.title(title)
    rmse = evaluate(target_var, validate, yhat)
    plt.legend()
    plt.show()
    
    
def append_eval_df(model_type, target_var, eval_df, validate, yhat):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, validate, yhat)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def run_models(train, val):
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    
    # Run last observed value model
    pred = train['avg_temp'][-1]
    yhat = pd.DataFrame({
    'avg_temp': [pred]},
    index = val.index)
    eval_df = append_eval_df(model_type='last_observed_value', target_var='avg_temp', eval_df=eval_df, validate=val, yhat=yhat)
    plot_and_eval('avg_temp', train, val, yhat, 'Last Observed Value Model')
    

    # Run simple average model
    pred = round(train['avg_temp'].mean(), 2)
    yhat = pd.DataFrame({
    'avg_temp': [pred]},
    index = val.index)
    eval_df = append_eval_df(model_type='simple_average', target_var='avg_temp', eval_df=eval_df, validate=val, yhat=yhat)
    plot_and_eval('avg_temp', train, val, yhat, 'Simple Average Model')

    
    # Run mimic of last year model
    y = train.tail(648)[['avg_temp']]
    yhat = y[['avg_temp']] + y[['avg_temp']].diff(365).mean()
    yhat.index = val.index
    eval_df = append_eval_df(model_type = "previous_year", 
                            target_var = 'avg_temp',
                        eval_df=eval_df, validate=val, yhat=yhat)
    plot_and_eval('avg_temp', train, val, yhat, 'Previous Year Model')

    
    return eval_df


def run_best_model(train, validate, test):
    y = train.tail(648)[['avg_temp']]
    yhat = y[['avg_temp']] + y[['avg_temp']].diff(365).mean()
    yhat.index = test.index
    plt.figure(figsize = (12,4))
    plt.plot(train['avg_temp'], label='Train', linewidth=1)
    plt.plot(validate['avg_temp'], label='Validate', linewidth=1)
    plt.plot(test['avg_temp'], label='Test', linewidth=1, color='red')
    plt.plot(yhat['avg_temp'], label='Prediction', linewidth=1, color='green')
    plt.title('Previous Year Model')
    plt.legend()
    plt.show()
    
    rmse = evaluate('avg_temp', validate, yhat)
    d = {'model_type': ['test_resutls'], 'target_var': ['avg_temp'],
        'rmse': [rmse]}
    
    return pd.DataFrame(d)