import numpy as np
import pandas as pd

def alpha_score(y_true,y_pred,alpha=1,beta=1,gamma=0.25):

    tp=np.sum(np.logical_and(y_true,y_pred))
    fn=np.sum(np.logical_and(y_true, np.logical_not(y_pred)))
    fp=np.sum(np.logical_and(np.logical_not(y_true),y_pred))

    alpha_num=(beta*fn + gamma*fp)
    alpha_den=(tp+fp+fn)

    a_score=1-(float(alpha_num)/alpha_den)

    return a_score
