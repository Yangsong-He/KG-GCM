import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

RESULTS_PT_DIR = os.path.join(CURRENT_DIR, "results")
RESULT_SAVE_DIR = os.path.join(CURRENT_DIR, "result")
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)


def evaluate_symbol(symbol: str):

    path = os.path.join(RESULTS_PT_DIR, f"{symbol}_results.pt")
    if not os.path.exists(path):
        print(f"[Eval] Prediction result file not found: {path}")
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)

    y_true = data["y_true"].numpy()
    y_pred = data["y_pred"].numpy()
    y_proba = data["y_proba"].numpy()
    best_val_acc = data.get("best_val_acc", None)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = None

    try:
        aupr = average_precision_score(y_true, y_proba)
    except Exception:
        aupr = None

    cm = confusion_matrix(y_true, y_pred)

    df_save = pd.DataFrame([{
        "symbol": symbol,
        "n_test": len(y_true),
        "best_val_acc": best_val_acc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "aupr": aupr,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }])

    save_path = os.path.join(RESULT_SAVE_DIR, f"{symbol}_metrics.csv")
    df_save.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[Eval] Metrics saved successfully: {save_path}")
    return df_save


SYMBOLS = [stock code]

def main():
    for sym in SYMBOLS:
        evaluate_symbol(sym)


if __name__ == "__main__":
    main()
