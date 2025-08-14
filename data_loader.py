import sys
import json
import os
with open("config.json", "r") as f:
    config = json.load(f)

sys.path.insert(0, config['RAUG_PATH'])

from raug.loader import get_data_loader
from img_aug import ImgTrainTransform, ImgEvalTransform
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from glob import glob

def get_paths_and_labels(path, lab_names):
    imgs_path, labels = list(), list()
    lab_cnt = 0
    for lab in lab_names:    
        paths_aux = glob(os.path.join(path, lab, "*.jpg"))
        
        labels += [lab_cnt] * len(paths_aux)
        imgs_path += paths_aux
        
        lab_cnt += 1
        
    return imgs_path, labels

def get_labels_name(data_path):
    labels_name = glob(os.path.join(data_path , "*"))
    labels_name = [l.split(os.path.sep)[-1] for l in labels_name]
    return labels_name

def create_data_loader(data_path, labels_name):

    img_path, labels = get_paths_and_labels(data_path, labels_name)


    train_imgs_paths, test_imgs_paths, train_labels, test_labels = train_test_split(img_path, labels, test_size=0.15, random_state=10)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_data_loader = []
    val_data_loader = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_imgs_paths, train_labels)):
        fold_train_paths = [train_imgs_paths[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]

        fold_val_paths = [train_imgs_paths[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        train_loader = get_data_loader(
            fold_train_paths, fold_train_labels,
            transform=ImgTrainTransform(),
            batch_size=30, shuf=True, pin_memory=True
        )

        val_loader = get_data_loader(
            fold_val_paths, fold_val_labels,
            transform=ImgEvalTransform(),
            batch_size=30, shuf=False, pin_memory=True
        )
        train_data_loader.append(train_loader)
        val_data_loader.append(val_loader)

    

    test_data_loader = get_data_loader(test_imgs_paths, test_labels, transform=ImgEvalTransform(),
                                        batch_size= 30, shuf=True, pin_memory=True)
    
    
    return train_data_loader, val_data_loader, test_data_loader