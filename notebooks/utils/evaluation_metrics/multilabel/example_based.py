import numpy as np
import pandas as pd


def hamming_loss(y_true,y_pred):
    
    hl_num=np.sum(np.logical_xor(y_true,y_pred))
    
    hl_den=np.prod(y_true.shape)
    
    return float(hl_num)/hl_den


def example_based_accuracy(y_true,y_pred):
    
    numerator=np.sum(np.logical_and(y_true,y_pred),axis=1)
    denominator=np.sum(np.logical_or(y_true,y_pred),axis=1)
    
    instance_accuracy=numerator/denominator
    avg_accuracy=np.mean(instance_accuracy)
    
    return avg_accuracy


def example_based_precision(y_true,y_pred):
    n=len(y_true)
    
    precision_num=np.sum(np.logical_and(y_true,y_pred),axis=1)
    precision_den=np.sum(y_pred,axis=1)

    p_n=np.array(precision_num).astype(np.float)
    p_d=np.array(precision_den).astype(np.float)

    avg_precision=np.nansum(p_n/p_d)/float(n)
    
    return avg_precision

def example_based_recall(y_true,y_pred):
    
    n=len(y_true)
    
    recall_num=np.sum(np.logical_and(y_true,y_pred),axis=1)
    recall_den=np.sum(y_true,axis=1)

    p_n=np.array(recall_num).astype(np.float)
    p_d=np.array(recall_den).astype(np.float)

    sum_recall=np.nansum(p_n/p_d)
    avg_recall=sum_recall/n
    
    return avg_recall
    