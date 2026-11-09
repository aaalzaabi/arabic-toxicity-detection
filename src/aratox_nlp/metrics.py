import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss

def multilabel_metrics(y_true, y_pred, labels):
    out = {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'hamming_loss': float(hamming_loss(y_true, y_pred)),
        'subset_accuracy': float(accuracy_score(y_true, y_pred)),
        'per_label': {}
    }
    pf = f1_score(y_true, y_pred, average=None, zero_division=0)
    pp = precision_score(y_true, y_pred, average=None, zero_division=0)
    pr = recall_score(y_true, y_pred, average=None, zero_division=0)
    for i, label in enumerate(labels):
        out['per_label'][label] = {'precision': float(pp[i]), 'recall': float(pr[i]), 'f1': float(pf[i]), 'support': int(y_true[:, i].sum())}
    return out

def binary_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'toxic_f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'precision_toxic': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_toxic': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

def threshold_predictions(scores, thresholds):
    return (np.asarray(scores) >= np.asarray(thresholds)).astype(int)
