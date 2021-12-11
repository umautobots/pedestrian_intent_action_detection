import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn import metrics

import pdb
def compute_AP(pred, target, info='', _type='action'):
    '''
    pred: (N, num_classes)
    target: (N)
    '''
    ignore_class = []
    class_index = ['standing', 'waiting',  'going towards', 
                    'crossing', 'crossed and standing', 'crossed and walking', 'other walking']
    # Compute AP
    result = {}
    for cls in range(len(class_index)):
        if cls not in ignore_class:
            result['AP '+class_index[cls]] = average_precision_score(
                                                    (target==cls).astype(np.int),
                                                    pred[:, cls])
            
            # print('{} AP: {:.4f}'.format(class_index[cls], result['AP n'+class_index[cls]]))

    # Compute mAP
    result['mAP'] = np.mean([v for v in result.values() if not np.isnan(v)])
    info += '\n'.join(['{}:{:.4f}'.format(k, v) for k, v in result.items()])
    return result, info

def compute_acc_F1(pred, target, info='', _type='action'):

    '''
    pred: (N, 1) or (N, 2)
    target: (N)
    '''
    result = {}
    if len(pred.shape) == 2:
        if pred.shape[-1] == 1:
            pred = np.round(pred[:, 0])
        elif pred.shape[-1] == 2:
            pred = np.round(pred[:, 1])
    else:
        pred = np.round(pred)
    acc_action = accuracy_score(target, pred)
    f1_action = f1_score(target, pred)
    precision = precision_score(target, pred)
    result[_type+'_accuracy'] = acc_action
    result[_type+'_f1'] = f1_action
    result[_type+'_precision'] = precision
    info += 'Acc: {:.4f}; F1: {:.4f}; Prec: {:.4f}; '.format(acc_action, f1_action, precision)
    return result, info

def compute_auc_ap(pred, target, info='', _type='action'):
    result = {}
    # NOTE: compute AUC
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    result[_type+'_auc'] = auc
    
    # NOTE: compute AP of crossing and not crossing and compute the mAP
    AP = average_precision_score(target, pred)
    result[_type+'_ap'] = AP
    info += 'AUC: {:.4f}; AP:{:.3f}; '.format(auc, AP)

    return result, info