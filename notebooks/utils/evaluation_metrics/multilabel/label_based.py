import numpy as np
import pandas as pd

# Macro evaluations

def accuracy_macro(y_true,y_pred):
    
    n=y_true.shape[1]
    
    numerator=np.sum(np.logical_and(y_true,y_pred),axis=0)
    denominator=np.sum(np.logical_or(y_true,y_pred),axis=0)
    
    
    p_n=np.array(numerator).astype(np.float)
    p_d=np.array(denominator).astype(np.float)
    
    sum_acc=np.nansum(p_n/p_d)
    avg_accuracy=sum_acc/n
    
    return avg_accuracy

def precision_macro(y_true,y_pred):
    
    n=y_true.shape[1]
    
    precision_num=np.sum(np.logical_and(y_true,y_pred),axis=0)
    precision_den=np.sum(y_pred,axis=0)

    p_n=np.array(precision_num).astype(np.float)
    p_d=np.array(precision_den).astype(np.float)

    sum_precision=np.nansum(p_n/p_d)
    avg_precision=sum_precision/n
    
    return avg_precision

def recall_macro(y_true,y_pred):
    
    n=y_true.shape[1]
    
    precision_num=np.sum(np.logical_and(y_true,y_pred),axis=0)
    precision_den=np.sum(y_true,axis=0)

    p_n=np.array(precision_num).astype(np.float)
    p_d=np.array(precision_den).astype(np.float)

    sum_recall=np.nansum(p_n/p_d)
    avg_recall=sum_recall/n
    
    return avg_recall

# Micro evaluations

def accuracy_micro(y_true,y_pred):
    
    numerator=np.sum(np.logical_and(y_true,y_pred))
    denominator=np.sum(np.logical_or(y_true,y_pred))
    
    p_n=np.array(numerator).astype(np.float)
    p_d=np.array(denominator).astype(np.float)
    
    micro_acc=(p_n/p_d)
    
    return micro_acc

def precision_micro(y_true,y_pred):
    
    precision_num=np.sum(np.logical_and(y_true,y_pred))
    precision_den=np.sum(y_pred)

    p_n=np.array(precision_num).astype(np.float)
    p_d=np.array(precision_den).astype(np.float)

    micro_precision=(p_n/p_d)
    
    return micro_precision

def recall_micro(y_true,y_pred):

    precision_num=np.sum(np.logical_and(y_true,y_pred))
    precision_den=np.sum(y_true)

    p_n=np.array(precision_num).astype(np.float)
    p_d=np.array(precision_den).astype(np.float)

    micro_recall=(p_n/p_d)
    
    return micro_recall