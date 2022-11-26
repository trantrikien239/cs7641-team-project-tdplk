import numpy as np
# Create cross validation set
from sklearn.model_selection import cross_validate
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split

def mcrmse(y_trues, y_preds):
    scores = []
    n_tasks = y_trues.shape[1]
    for i in range(n_tasks):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def mcrmse_error(y_trues, y_preds, **kwargs):
    mcrmse_score, _ = mcrmse(y_trues, y_preds)
    return mcrmse_score


mcrmse_scorer = make_scorer(mcrmse_error, greater_is_better=False)