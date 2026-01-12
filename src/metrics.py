from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_classification(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
