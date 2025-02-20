"""
The rainymotion library provides different goodness of fit metrics for
nowcasting models' performance evaluation.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
from typing import List
from functools import partial

import numpy as np
import meteva.method as mem
import meteva.product as mpd
import meteva.base as meb

import xarray as xr

# -- Regression metrics -- #


def R(obs, sim):
    """
    Correlation coefficient

    Reference:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: correlation coefficient between observed and simulated values

    """
    obs = obs.flatten()
    sim = sim.flatten()

    return np.corrcoef(obs, sim)[0, 1]


def R2(obs, sim, axis=None):
    """
    Coefficient of determination

    Reference:
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: coefficient of determination between observed and
               simulated values

    """

    # obs = obs.flatten()
    # sim = sim.flatten()

    numerator = np.sum(((obs - sim) ** 2), axis = axis)

    denominator = np.sum(((obs - np.mean(obs)) ** 2), axis = axis)

    return 1 - numerator/denominator


def RMSE(obs, sim, axis =None):
    """
    Root mean squared error

    Reference: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: root mean squared error between observed and simulated values

    """
    # obs = obs.flatten()
    # sim = sim.flatten()

    return np.nanmean(np.sqrt(np.mean((obs - sim) ** 2)), axis = axis)

def ME(obs, sim ,axis =None):
    """
    Mean absolute error

    Reference: https://en.wikipedia.org/wiki/Mean_absolute_error

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: mean absolute error between observed and simulated values

    """
    # obs = obs.flatten()
    # sim = sim.flatten()

    return np.abs(sim - obs)

def MAE(obs, sim, axis =None):
    """
    Mean absolute error

    Reference: https://en.wikipedia.org/wiki/Mean_absolute_error

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations

    Returns:
        float: mean absolute error between observed and simulated values

    """
    # obs = obs.flatten()
    # sim = sim.flatten()

    return np.nanmean(np.abs(sim - obs), axis = axis)

# -- Radar-specific classification metrics -- #


def prep_clf_bak(obs, sim, threshold=0.1, compare = '>=' ,axis =None):
    '''

    :param obs:
    :param sim:
    :param threshold:
    :param compare:
    :param axis:
    :return:
    '''

    if compare not in [">=",">","<","<="]:
        print("compare 参数只能是 >=   >  <  <=  中的一种")
        return
    if compare == ">=":
        obs = np.where(obs >= threshold, 1, 0)
        sim = np.where(sim >= threshold, 1, 0)
    elif compare == "<=":
        obs = np.where(obs <= threshold, 1, 0)
        sim = np.where(sim <= threshold, 1, 0)
    elif compare == ">":
        obs = np.where(obs > threshold, 1, 0)
        sim = np.where(sim > threshold, 1, 0)
    elif compare == "<":
        obs = np.where(obs < threshold, 1, 0)
        sim = np.where(sim < threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (sim == 1), axis = axis)

    # False negative (FN)
    misses = np.sum((obs == 1) & (sim == 0), axis = axis)

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (sim == 1), axis = axis)

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (sim == 0), axis = axis)

    return hits, misses, falsealarms, correctnegatives

def prep_clf(obs: np.ndarray , sim: np.ndarray, grade_list: List=None,
             compare: str='>=', axis=None, return_array: bool=False):
    '''

    :param obs:
    :param sim:
    :param threshold:
    :param compare:
    :param axis:
    :return:
    '''

    if compare not in [">=",">","<","<="]:
        print("compare 参数只能是 >=   >  <  <=  中的一种")
        return
    if obs.shape != sim.shape:
        print('预报数据和观测数据维度不匹配')
        return

    if grade_list is None:
        grade_list = [1e-30]

    #Ob_shape = [obs[0,:,:].shape if axis==0 else (1,) ][0] # axis=0 == (lat,lon) | axis=(1,2) == t | axis=None == (1)
    Ob_shape = [obs.shape[-2:] if axis==0 else (1,) ][0] # axis=0 == (lat,lon) | axis=(1,2) == t | axis=None == (1)

    hfmc_array = np.zeros((len(grade_list), 4, *Ob_shape))
    print('>>>> (threshold, 混淆矩阵, lat, lon) = ',hfmc_array.shape)
    for i in range(len(grade_list)):
        threshold = grade_list[i]
        if compare == ">=":
            obs = np.where(obs >= threshold, 1, 0)
            sim = np.where(sim >= threshold, 1, 0)
        elif compare == "<=":
            obs = np.where(obs <= threshold, 1, 0)
            sim = np.where(sim <= threshold, 1, 0)
        elif compare == ">":
            obs = np.where(obs > threshold, 1, 0)
            sim = np.where(sim > threshold, 1, 0)
        elif compare == "<":
            obs = np.where(obs < threshold, 1, 0)
            sim = np.where(sim < threshold, 1, 0)

        # True positive (TP)
        '''
        hits = np.sum((obs == 1) & (sim == 1), axis = axis)
        # False negative (FN)
        misses = np.sum((obs == 1) & (sim == 0), axis = axis)
        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (sim == 1), axis = axis)
        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (sim == 0), axis = axis)
        '''
        hits = np.where((obs == 1) & (sim == 1), 1, 0)
        # False negative (FN)
        misses = np.where((obs == 1) & (sim == 0), 1, 0)
        # False positive (FP)
        falsealarms = np.where((obs == 0) & (sim == 1), 1, 0)
        # True negative (TN)
        correctnegatives = np.where((obs == 0) & (sim == 0), 1, 0)

        if not return_array:
            hits = np.sum(hits, axis=axis)
            misses = np.sum(misses, axis=axis)
            falsealarms = np.sum(falsealarms, axis=axis)
            correctnegatives = np.sum(correctnegatives, axis=axis)

        # hfmc_array.append(hits)
        # hfmc_array.append(misses)
        # hfmc_array.append(falsealarms)
        # hfmc_array.append(correctnegatives)
        hfmc_array[i, 0, :] = hits
        hfmc_array[i, 1, :] = misses
        hfmc_array[i, 2, :] = falsealarms
        hfmc_array[i, 3, :] = correctnegatives

    Hits, Misses, Falsealarms, Correctnegatives =  hfmc_array[:,0, :], hfmc_array[:,1, :], hfmc_array[:,2, :], hfmc_array[:,3, :]

    #return Hits, Misses, Falsealarms, Correctnegatives
    res = np.array([Hits, Misses, Falsealarms, Correctnegatives])
    return res

def BIAS(obs, sim , grade_list : List =[1e-30], compare:str = '>=' ,axis =None, return_array=False):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
          alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    return (hits + falsealarms) / (hits + misses)


def CSI(obs : np.ndarray, sim:np.ndarray , grade_list : List =[1e-30], compare:str = '>=' ,axis =None, return_array=False):
    """
    CSI - critical success index

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: CSI value

    """

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    return hits / (hits + misses + falsealarms)


def FAR(obs:np.ndarray, sim:np.ndarray , grade_list : List =[1e-30], compare:str = '>=' ,axis =None, return_array=False):
    '''
    FAR - false alarm rate

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: FAR value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    return falsealarms / (hits + falsealarms)

FAR_R = partial(FAR, compare='<')

def FAR_R_zym(obs, sim, threshold=0.1, axis = None):
    '''
    FAR - false alarm rate

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: FAR value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf_bak(obs=obs, sim=sim,
                                                           threshold=threshold, axis = axis)

    return misses / (misses + correctnegatives)

def POD(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    POD - probability of detection

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: POD value

    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    return hits / (hits + misses)

def POD_R_zym(obs, sim, threshold=0.1, axis = None):
    '''
    POD - probability of detection

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: POD value

    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim=sim,
                                                           threshold=threshold, axis = axis)

    return correctnegatives / (falsealarms + correctnegatives)

POD_R = partial(POD, compare='<')

def HSS(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    HSS - Heidke skill score

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: HSS value

    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den

def KSS(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    KSS -

    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: HSS value

    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    KSS_num = (hits * correctnegatives) - misses - falsealarms
    KSS_den = (hits + misses) * (falsealarms + correctnegatives)

    return KSS_num / KSS_den


def ETS(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    ETS - Equitable Threat Score
    details in the paper:
    Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
    radar-derived precipitation with model-derived winds.
    Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: ETS value

    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

    return ETS


def BSS(obs :np.ndarray, sim:np.ndarray, threshold=0.1):
    '''
    BSS - Brier skill score

    details:
    https://en.wikipedia.org/wiki/Brier_score

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: BSS value

    '''
    obs = np.where(obs >= threshold, 1, 0)
    sim = np.where(sim >= threshold, 1, 0)

    obs = obs.flatten()
    sim = sim.flatten()

    return np.sqrt(np.mean((obs - sim) ** 2))

# -- ML-specific classification metrics -- #


def ACC(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    ACC - accuracy score

    details:
    https://en.wikipedia.org/wiki/Accuracy_and_precision

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: accuracy value

    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    return (TP + TN) / (TP + TN + FP + FN)



def precision(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    precision - precision score

    details:
    https://en.wikipedia.org/wiki/Information_retrieval#Precision

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: precision value

    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim=sim, grade_list=grade_list, compare=compare,
                                                               axis=axis, return_array=return_array)
    return TP / (TP + FP)


def recall(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    recall - recall score

    details:
    https://en.wikipedia.org/wiki/Information_retrieval#Recall

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: recall value
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    return TP / (TP + FN)


def FSC(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None):
    '''
    FSC - F-score

    details:
    https://en.wikipedia.org/wiki/F1_score

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: FSC value
    '''

    pre = precision(obs, sim, grade_list, compare, axis)
    rec = recall(obs, sim, grade_list, compare, axis)

    return 2 * ((pre * rec) / (pre + rec))


def MCC(obs:np.ndarray, sim:np.ndarray, grade_list=[1e-30], compare:str = '>=', axis = None, return_array=False):
    '''
    MCC - Matthews correlation coefficient

    details:
    https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

    Args:
        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)

    Returns:
        float: MCC value
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, sim = sim, grade_list = grade_list, compare = compare, axis = axis, return_array=return_array)

    MCC_num = TP * TN - FP * FN
    MCC_den = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))

    return MCC_num / MCC_den

# -- Curves for plotting -- #


def ROC_curve(obs, sim, thresholds, axis = None):
    '''
    ROC - Receiver operating characteristic curve coordinates

    Reference: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

    Args:

        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        thresholds (list with floats): number of thresholds over which
                                       we consider rain falls

    Returns:

        tpr (numpy.ndarray): true positive rate according to selected
                             thresholds (y axis on ROC)
        fpr (numpy.ndarray): false positive rate according to selected
                             thresholds (x axis on ROC)

    '''

    tpr = []
    fpr = []

    for threshold in thresholds:

        TP, FN, FP, TN = prep_clf_bak(obs=obs, sim=sim, threshold=threshold, axis = axis)

        tpr.append(TP / (TP + FN))

        fpr.append(FP / (FP + TN))

    return np.array(tpr), np.array(fpr)


def PR_curve(obs, sim, thresholds, axis = None):
    '''
    PRC - precision-recall curve coordinates

    Reference:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

    Args:

        obs (numpy.ndarray): observations
        sim (numpy.ndarray): simulations
        thresholds (list with floats): number of thresholds over which
                                       we consider rain falls

    Returns:

        pre (numpy.ndarray): precision rate according to selected thresholds
                             (y axis on PR)
        rec (numpy.ndarray): recall rate according to selected thresholds
                             (x axis on PR)

    '''

    pre = []
    rec = []

    for threshold in thresholds:

        pre.append(precision(obs=obs, sim=sim, threshold=threshold, axis = axis))
        rec.append(recall(obs=obs, sim=sim, threshold=threshold, axis = axis))

    return np.array(pre), np.array(rec)


def AUC(x, y):
    '''
    AUC - area under curve

    Note: area under curve wich had been computed by standard trapezial
          method (np.trapz)

    Args:

        x (numpy.ndarray): array of one metric rate (1D)
        y (numpy.ndarray): array of another metric rate (1D)

    Returns:

        float - area under curve

    '''

    return np.trapz(y, x)


def agg_TS(confusion_array):
    '''Compute TS score by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): TS score defined as TS = hits / (hits + misses + falsealarms)
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)

    res =  hits / (hits + misses + falsealarms)

    return res


def agg_BIAS(confusion_array):
    '''Compute bias by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): bias score defined as bias = (hits + falsealarms) / (hits + misses )
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)

    return (hits + falsealarms) / (hits + misses)


def agg_FAR(confusion_array):
    '''Compute FAR by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): FAR score defined as FAR = (hits + falsealarms) / (hits + misses)
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)

    return (hits + falsealarms) / (hits + misses)


def agg_POD(confusion_array):
    '''Compute POD by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): POD score defined as FAR = hits / (hits + misses)
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    #falsealarms = confusion_array[:, 2, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    #falsealarms = np.nansum(falsealarms, axis=0)

    return hits / (hits + misses)


def agg_HSS(confusion_array):
    '''Compute HSS by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): HSS score.
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]
    correctnegatives = confusion_array[:, 3, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)
    correctnegatives = np.nansum(correctnegatives, axis=0)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den


def agg_KSS(confusion_array):
    '''Compute KSS by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): KSS score.
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]
    correctnegatives = confusion_array[:, 3, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)
    correctnegatives = np.nansum(correctnegatives, axis=0)

    KSS_num = (hits * correctnegatives) - misses - falsealarms
    KSS_den = (hits + misses) * (falsealarms + correctnegatives)

    return KSS_num / KSS_den


def agg_ETS(confusion_array):
    '''Compute ETS by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): ETS score.
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]
    correctnegatives = confusion_array[:, 3, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)
    correctnegatives = np.nansum(correctnegatives, axis=0)

    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    res = (hits - Dr) / (hits + misses + falsealarms - Dr)

    return res


def agg_ACC(confusion_array):
    '''Compute ACC by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): ACC score.
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]
    correctnegatives = confusion_array[:, 3, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)
    correctnegatives = np.nansum(correctnegatives, axis=0)

    res = (hits + correctnegatives) / (hits + misses + falsealarms + correctnegatives)

    return res


def agg_precision(confusion_array):
    '''Compute precision by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): precision score.
    '''

    hits = confusion_array[:, 0, ...]
    #misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]
    #correctnegatives = confusion_array[:, 3, ...]

    hits = np.nansum(hits, axis=0)
    #misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)
    #correctnegatives = np.nansum(correctnegatives, axis=0)

    res = hits / (hits + falsealarms)

    return res


agg_recall = agg_POD


def agg_MCC(confusion_array):
    '''Compute MCC by aggregating confusion table counts over a period

    Args:
        confusion_array (ndarray): ndarray with shape (time, 4, ...).
        1st dimension is time, over which the total counts are summed.
        2nd dimension is 4, and are the hits, misses, falsealarms, correctnegatives
        counts in a confusion table.
    Returns:
        res (ndarray): MCC score.
    '''

    hits = confusion_array[:, 0, ...]
    misses = confusion_array[:, 1, ...]
    falsealarms = confusion_array[:, 2, ...]
    correctnegatives = confusion_array[:, 3, ...]

    hits = np.nansum(hits, axis=0)
    misses = np.nansum(misses, axis=0)
    falsealarms = np.nansum(falsealarms, axis=0)
    correctnegatives = np.nansum(correctnegatives, axis=0)

    MCC_num = hits * correctnegatives - falsealarms * misses
    MCC_den = np.sqrt((hits + falsealarms)*(hits + misses)*(correctnegatives + falsealarms)*(correctnegatives + misses))

    return MCC_num / MCC_den


def MODE(ob, fo, smooth = 5, threshold = 5, minsize = 5, compare = ">=", save_dir = None, cmap = "rain_24h", clevs = None, show = False):
    ''' mode 检验方法

    :param grd_ob: xr.DataArray 格式数据
    :param grd_fo: ,xr.DataArray 格式数据
    :param smooth: 平滑系数， 为0不平滑
    :param threshold: 阈值，平滑后低于阈值被置0
    :param minsize: 最小像素点面积，小于该值沪忽略不计
    :param compare: 比较方法，可选项包括”>=”,”>”,”<=”,”<”
    :param save_dir:

    :return: 混交矩阵字典

    '''
    grd_ob = meb.xarray_to_griddata(ob)
    grd_fo = meb.xarray_to_griddata(fo)
    # 判断是否格式正确
    if grd_ob.dims == grd_fo.dims:
        feature = mem.mode.operate(grd_ob, grd_fo, smooth = smooth, threshold = threshold, minsize = minsize, compare = compare,
                                   save_dir = save_dir, cmap = cmap, clevs = clevs, show = show)
        return feature['feature_table']['contingency_table_yesorno']
    else:
        Exception('grd_ob or grd_fo data is erro !!!')


def CRA(ob, fo, smooth = 5, threshold = 5, minsize = 5, compare = ">="):
    ''' CRA 目标检验
        (meteva >= 1.8)
    :param ob:
    :param fo:
    :param smooth:
    :param threshold:
    :param minsize:
    :param compare:
    :return:

    '''
    out = {}
    dict_name = ['质心经度偏差（预报-观测）','质心纬度偏差（预报-观测）','目标倾角偏差（预报-观测）','观测均值','预报均值','总误差','平移场残差','平移误差','平移旋转场残差',
                 '旋转误差','强度误差','形态误差']
    var_name = ['delta_lon', 'delta_lat', 'delta_angle', 'ob_mean', 'fo_mean', 'MSE_total', 'MSE_shift_left', 'MSE_shift', 'MSE_shift_rotate_left',
                        'MSE_rotate', 'MSE_volume', 'MSE_pattern']
    grd_ob = meb.xarray_to_griddata(ob)
    grd_fo = meb.xarray_to_griddata(fo)
    # 判断是否格式正确
    if grd_ob.dims == grd_fo.dims:
        look_featureFinder = mem.mode.feature_finder(grd_ob, grd_fo, smooth=smooth, threshold=threshold, minsize=minsize, compare=compare)
        look_match = mem.mode.centmatch(look_featureFinder)
        look_merge = mem.mode.merge_force(look_match)
        look_cra = mem.cra.craer(look_merge, stages=True, translate=True, rotate=True)
        if len(look_cra.keys()) == 0:
            print('无匹配上的目标！！！')
            out['成功匹配目标数量'] = 0
            return out
        else:
            out['成功匹配目标数量'] = len(look_cra.keys())
            for k, var in zip(dict_name, var_name):
                out[k] = np.mean([look_cra[i][var] for i in look_cra.keys()])

        return out


    else:
        Exception('grd_ob or grd_fo data is erro !!!')


if __name__ == '__main__':
    # obs = np.random.randint(0, 4, size=(100, 50, 50))
    # sim = np.random.randint(0, 4, size=(100, 50, 50))
    #
    # pod_score = POD(obs, sim, grade_list=[1], compare='>=', axis=None)
    # print(pod_score)
    # Hits, Misses, Falsealarms, Correctnegatives = prep_clf_new(obs, sim, grade_list=[1], compare='>=', axis=0)
    # print('Hits:',Hits.shape, 'therold :' ,[0,1,2])
    # pod = Correctnegatives / (Falsealarms + Correctnegatives)
    # print(pod[0])


    # ========= MODE and CRA ==============
    folder = os.path.join(Path(os.path.abspath(__file__)).parent.parent, 'data', 'MODE&CRA')
    save_dir = os.path.join(folder, 'res')
    #

    obs = xr.open_dataset(os.path.join(folder, 'rain3h_012.nc'))
    fo  = xr.open_dataset(os.path.join(folder, 'rain3h_2023060112.nc'))
    #
    feature = MODE(ob = obs, fo = fo , smooth = 5, threshold = 5, minsize = 5, compare = ">=",
         save_dir = save_dir, cmap = "rain_24h", clevs = None, show = False)
    print('MODE 混交矩阵：',feature)

    cra_dict = CRA(ob = obs, fo = fo , smooth=1, threshold=5, minsize=5, compare=">=")
    print('CRA : ', cra_dict)



