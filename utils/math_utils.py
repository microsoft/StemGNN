

import numpy as np
import math
from scipy.stats import pearsonr
import pandas as pd

# #def z_score(x, mean, std):
#     '''
#     Z-score normalization function: $z = (X - \mu) / \sigma $,
#     where z is the z-score, X is the value of the element,
#     $\mu$ is the population mean, and $\sigma$ is the standard deviation.
#     :param x: np.ndarray, input array to be normalized.
#     :param mean: float, the value of mean.
#     :param std: float, the value of standard deviation.
#     :return: np.ndarray, z-score normalized array.
#     '''
# #    return (x - mean) / std

def z_score(x, x_stats):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.

    '''

    for i in range(0,len(x[0])):
        x[:,i]=(x[:,i]-x_stats[i]['mean'])/x_stats[i]['std']
    
    x[np.where(x == 0)] = np.nan
    x = pd.DataFrame(x)
    x = x.fillna(method='ffill', limit=len(x)).fillna(method='bfill', limit=len(x))
    x = np.asarray(x.values)
    
    return x



def z_inverse(x, x_stats):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    if len(x.shape)>=2:
        for i in range(0,len(x[0])):
            x[:,i]=(x[:,i]*x_stats[i]['std'])+ x_stats[i]['mean']
    else:
        for i in range(0,len(x[0])):
            x[i]=(x[i]*x_stats[i]['std'])+ x_stats[i]['mean']

    return x

# def z_inverse(x, mean, std):
#     '''
#     The inverse of function z_score().
#     :param x: np.ndarray, input to be recovered.
#     :param mean: float, the value of mean.
#     :param std: float, the value of standard deviation.
#     :return: np.ndarray, z-score inverse array.
#     '''
#     for i in range(0,len(x[0])):
#         x[:,i]=(x[:,i]*x_stats[i]['std'])+ x_stats[i]['mean']
#     return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.abs(np.mean(np.abs(v_ - v) / v)


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))

def rrse_(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def CORR(y_true, y_pred):
    y_true = np.array(y_true, 'float32')
    y_pred = np.array(y_pred, 'float32')
    N = y_true.shape[0]

    total = 0.0
    for i in range(N):
        print(y_true[i])
        if math.isnan(pearsonr(np.array(y_true[i], 'float32'), np.array(y_pred[i], 'float32'))[0]):

            N -= 1
        else:
            total += pearsonr(np.array(y_true[i], 'float32'), np.array(y_pred[i], 'float32'))[0]
    return total / N


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)

    #print(y)
    #print(y_)

    if dim == 3:
        # single_step case

        v = y#z_inverse(np.squeeze(y), x_stats)
        v_ = z_inverse(y_, x_stats)
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
        #return np.array([rrse_(v, v_), MAE(v, v_), CORR(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
