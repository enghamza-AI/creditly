
import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
   
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))  
    fn = np.sum((y_true == 1) & (y_pred == 0))  
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    c = confusion_counts(y_true, y_pred)
    eps = 1e-15  
    precision = c["tp"] / (c["tp"] + c["fp"] + eps)
    recall = c["tp"] / (c["tp"] + c["fn"] + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc_manual(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
 
    order = np.argsort(y_pred_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_pred_prob) + 1)   

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC undefined with only one class present in y_true.")

    sum_ranks_pos = np.sum(ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc
