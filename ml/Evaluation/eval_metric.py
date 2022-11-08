import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")

## Calculating MCMRSE
def mcrmse(data,actual_cols, pred_cols): 
    mse_list=[]
    for i in range(0,len(actual_cols)):
        mse = mean_squared_error(data[actual_cols[i]], data[pred_cols[i]])
        mse_list.append(mse)
    return np.mean(mse_list)

## rounding the number to closest multiple of "Multiple" input
def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

## Calculating Quadratic Weighted Kappa by transforming contiunous to categorical response
def quad_kappa(data,actual_cols, pred_cols):
    kappa_list=[]
    for i in range(0,len(actual_cols)):
        dummy = data[[actual_cols[i],pred_cols[i]]]
        dummy[actual_cols[i]] = dummy[actual_cols[i]].apply(lambda x: str(round_to_multiple(x,0.5)).strip())
        dummy[pred_cols[i]] = dummy[pred_cols[i]].apply(lambda x: str(round_to_multiple(x,0.5)).strip())
        quad_kappa = cohen_kappa_score(dummy[actual_cols[i]],
                                       dummy[pred_cols[i]], 
                                       labels=None, weights= 'quadratic', sample_weight=None)
        kappa_list.append(quad_kappa)
    return kappa_list


