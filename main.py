import sys
import json
import os
with open("config.json", "r") as f:
    config = json.load(f)

sys.path.insert(0, config['RAUG_PATH'])

from raug.loader import get_data_loader
from raug.checkpoints import save_model_as_onnx
from raug.train import fit_model
from raug.eval import test_model
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import time
from glob import glob
from sacred import Experiment
from sacred.observers import FileStorageObserver
import numpy as np
from data_loader import get_labels_name,  create_data_loader
from models.load_model import set_model
from cross_validation_metrics import get_cross_validation_metrics

#Params
NUM_EPOCHS = 50
model_name = 'mobilenet'
lr = 0.00001



DATA_PATH = config['DATA_PATH']
RESULT_PATH  = "results/" + "_" + model_name + "_fold_"  + str(time.time()).replace('.', '')

_metric_options_test = {
        'save_all_path': os.path.join(RESULT_PATH, "test_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}





labels_name = get_labels_name(DATA_PATH)

train_data_loader, val_data_loader, test_data_loader = create_data_loader(DATA_PATH, labels_name=labels_name)


for i in range(0,5):
    
    model = set_model(model_name, num_class=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    log_path = os.path.join(RESULT_PATH, "folder" + str(i), "log")
    
    fit_model(model, train_data_loader[i], val_data_loader[i], optimizer=optimizer, loss_fn=loss_fn, epochs=NUM_EPOCHS, save_folder=log_path)
    
    _metric_options_train = {
        'save_all_path': os.path.join(RESULT_PATH, "folder" + str(i), "train_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    
    _metric_options = {
        'save_all_path': os.path.join(RESULT_PATH, "folder" + str(i), "val_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    
    test_model (model, train_data_loader[i], loss_fn=loss_fn, save_pred=True,
                metrics_to_comp='all', class_names=labels_name, metrics_options=_metric_options_train,
                apply_softmax=True, verbose=False)
    
    test_model (model, val_data_loader[i], loss_fn=loss_fn, save_pred=True,
                metrics_to_comp='all', class_names=labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    



test_model (model, test_data_loader, loss_fn=loss_fn, save_pred=True,
                metrics_to_comp='all', class_names=labels_name, metrics_options=_metric_options_test,
                apply_softmax=True, verbose=False)


get_cross_validation_metrics(RESULT_PATH, label_names=labels_name)