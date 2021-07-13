from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import log_loss
import numpy as np


def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)  
    return sum(squaredError) / len(squaredError)


def MSE_ips(preds, true, item, user_num, item_num, inverse_propensity):
    squaredError = []
    globalNormalizer = 0
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(
            dis * dis * inverse_propensity[item[i]-1][int(true[i]) - 1])
        globalNormalizer += inverse_propensity[item[i]-1][int(true[i]) - 1]
    return sum(squaredError) / len(squaredError)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        absError.append(abs(dis)) 
    return sum(absError) / len(absError)


def MAE_ips(preds, true, item, user_num, item_num, inverse_propensity):
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        absError.append(
            abs(dis) * inverse_propensity[item[i]-1][int(true[i]) - 1])
    return sum(absError) / len(absError)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
        absError.append(abs(dis))
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def RMSE_ips(preds, true, item, user_num, item_num, inverse_propensity):
    squaredError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(
            dis * dis * inverse_propensity[item[i]-1][int(true[i]) - 1])
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def Acc(true, preds):
    return accuracy_score(true, preds)


def AUC(true, preds):
    return roc_auc_score(true, preds)


def NLL(true, preds):
    return -log_loss(true, preds, eps=1e-7)

def AUC_matrix(true, pred):

    pos = pred[np.where(true==1)]
    neg = pred[np.where(true==0)]
    I = np.zeros([len(pos), len(neg)])

    for i, x in enumerate(pos):
        for j, y in enumerate(neg):
            if x > y:
                I[i, j] = 1
            elif x==y:
                I[i, j] = 0.5
            else:
                I[i, j] = 0
    
    auc = np.mean(I)

    return auc
