import pandas as pd
import numpy as np
import sys
import json
import os
with open("config.json", "r") as f:
    config = json.load(f)

sys.path.insert(0, config['RAUG_PATH'])
from raug.utils import classification_metrics as cmet

def get_cross_validation_metrics(path, label_names = None):
    acc = _get_cross_validation_acurracy(path)
    bal_acc = _get_cross_validation_balenced_accurracy(path)
    pre_rec = _get_precision_recall_cross_validation(path, label_names)
    file = os.path.join(path, "cross_val_metrics", "metrics.txt")
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Balanced accuracy: {bal_acc}\n")
        f.write(f"{pre_rec}\n")
    _get_confusion_matrix_cross_validation(path, label_names=label_names)

def _get_cross_validation_acurracy(path):
    real = np.array([])
    pred = np.array([])
    for i in range(0,5):
        file_path = os.path.join(path, "folder" + str(i+1), "val_metrics","predictions_best_test.csv")
        arq = pd.read_csv(file_path)
        real = np.append(real, arq['REAL'])
        pred = np.append(pred, arq['PRED'])
        
    return  cmet.accuracy(real, pred)

def _get_cross_validation_balenced_accurracy(path):
    real = np.array([])
    pred = np.array([])
    for i in range(0,5):
        file_path = os.path.join(path, "folder" + str(i+1), "val_metrics","predictions_best_test.csv")
        arq = pd.read_csv(file_path)
        real = np.append(real, arq['REAL'])
        pred = np.append(pred, arq['PRED'])
        
    return  cmet.balanced_accuracy(real, pred)

def _get_precision_recall_cross_validation(path, label_names):
    real = np.array([])
    pred = np.array([])
    for i in range(0,5):
        file_path = os.path.join(path, "folder" + str(i+1), "val_metrics","predictions_best_test.csv")
        arq = pd.read_csv(file_path)
        real = np.append(real, arq['REAL'])
        pred = np.append(pred, arq['PRED'])
        
    return  cmet.precision_recall_report(real, pred, class_names = label_names)

def _get_confusion_matrix_cross_validation(path, label_names):
    real = np.array([])
    pred = np.array([])
    for i in range(0,5):
        file_path = os.path.join(path, "folder" + str(i+1), "val_metrics","predictions_best_test.csv")
        arq = pd.read_csv(file_path)
        real = np.append(real, arq['REAL'])
        pred = np.append(pred, arq['PRED'])
        
    cm = cmet.conf_matrix(real, pred)
    cm_path = os.path.join(path, "cross_val_metrics", "conf_mat")
    cmet.plot_conf_matrix(cm, label_names, save_path=cm_path)

