import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt
import seaborn as sns

# working with dates
from datetime import datetime

# statistics
from scipy import stats



def get_q1_vis(train):
    train.resample('5Y').mean().avg_temp.plot(title='5 Year Average Temperatures Stay Steady',
                                         xlabel='Year',
                                         ylabel='Temperature (F)',
                                         figsize=(12,8))
    plt.legend()
    plt.show()
    
    train.resample('Y').mean().rolling(5).mean().avg_temp.plot(title='Rolling 5-year Average Temperatures Show Similiar Trend',
                                         xlabel='Year',
                                         ylabel='Temperature (F)',
                                         figsize=(12,8))
    plt.legend()
    plt.show()
    
def get_q1_stats(train):
    before = train[:'1820'].avg_temp
    after = train['1820':].avg_temp
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. Findings suggest the average temperature before 1820 is different than after.')
    else:
        print('Fail to reject the Null Hypothesis. Findings suggest the average temperature before 1820 is the same as after.')
        
        
def get_q2_vis(train):
    train.resample('2Y').mean().temp_uncertainty.plot(title='Uncertainty Drops Over Time',
                                         xlabel='Year',
                                         ylabel='Temperature (F)',
                                         figsize=(12,8))
    plt.legend()
    plt.show()
    
def get_q2_stats(train):
    before = train[:'1820'].temp_uncertainty
    after = train['1820':].temp_uncertainty
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α and t > 0:
        print('Reject the Null Hypothesis. Findings suggest the temperature uncertainty before 1820 is higher than after.')
    else:
        print('Fail to reject the Null Hypothesis. Findings suggest the temperature uncertainty before 1820 is the same or less than as after.')
        
        
def get_q3_vis(train):
    y = train[1720:].avg_temp
    stack = y.groupby([y.index.year, y.index.month]).mean().unstack(0) # 0 or 1 will change how its unstacked
    stack.plot(title='Monthly Seasonal Plot', xlabel='Month', ylabel='Temperature', figsize = (12,8), cmap='Blues')
    plt.legend()
    plt.show()
    y = train[1820:].resample('3M').mean().avg_temp
    stack = y.groupby([y.index.year, y.index.month]).mean().unstack(0) # 0 or 1 will change how its unstacked
    stack.plot(title='Quarterly Seasonal Plot', xlabel='Quarter', ylabel='Temperature', figsize = (12,8), cmap='Blues')
    plt.legend()
    plt.show()
    

