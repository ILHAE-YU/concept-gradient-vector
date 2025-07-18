import os
import sys
from tqdm import tqdm
from itertools import product
from joblib import Parallel, delayed
import joblib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath('./lib'))

from datamodule.DaTdataset import DaTdataset, MinMaxNormalization, SBRMeanStdNormalization, calc_sbr_mean_std
from datamodule.helper import split_data, feature_extract
from config import *

# data split
train_data, train_labels, prototype_data, prototype_labels, test_data, test_labels, excluded_data, excluded_labels = split_data(
    TABLE_PATH,
    DATA_PATH,
    SBR_COLUMNS,
    SPLIT_RATIO,
    EXCLUDE,
    RANDOM_SEED,
    TEST_FIX_SEED
)

# setup
train_means = []
train_stds = []

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for n_fold, (train_idx, valid_idx) in enumerate(k_fold.split(train_data, train_labels)):
    if n_fold != TEST_FOLD:
        continue

    X_train, X_valid = [train_data[i] for i in train_idx], [train_data[i] for i in valid_idx]
    y_train, y_valid = [train_labels[i] for i in train_idx], [train_labels[i] for i in valid_idx]

    train_mean, train_std = calc_sbr_mean_std(X_train)
    train_means.append(train_mean.cpu().numpy())
    train_stds.append(train_std.cpu().numpy())

    transform = transforms.Compose([
        transforms.ToTensor(),
        MinMaxNormalization(0.0, 1.0),
        transforms.Lambda(lambda x: x.type(torch.float32)),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Lambda(lambda x: x[..., 13:-14, 13:-14, 6:-7])
    ])

    sbr_transform = transforms.Compose([
        SBRMeanStdNormalization(train_mean, train_std)
    ])

    train_dataset = DaTdataset(X_train, y_train, transform, sbr_transform)
    valid_dataset = DaTdataset(X_valid, y_valid, transform, sbr_transform)
    prototype_dataset = DaTdataset(prototype_data, prototype_labels, transform, sbr_transform)
    test_dataset = DaTdataset(test_data, test_labels, transform, sbr_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    prototype_loader = DataLoader(prototype_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

base_dir = os.path.join(STUDY_NAME, f"fold_{TEST_FOLD}")
model = MODEL(num_classes=1).cuda()
model.load_state_dict(torch.load(os.path.join(base_dir, "best.pth")))
model.eval()

prototype_featuremaps, prototype_sbrs, valid_featuremaps, valid_sbrs, test_featuremaps, test_sbrs, feature_maps_length = feature_extract(
    model, prototype_loader, valid_loader, test_loader
)

svrs = []

os.makedirs(os.path.join(base_dir, "cgv"), exist_ok=True)
SAVE_PATH = os.path.join(STUDY_NAME, f"fold_{TEST_FOLD}", 'cgv')

for fmap_idx in range(feature_maps_length):
    svrs.append([])
    for i in tqdm(range(len(SBR_COLUMNS)), desc=f"{fmap_idx}-svrs train", leave=False):
        
        x = prototype_featuremaps[fmap_idx].cpu().numpy()
        y = prototype_sbrs[:, i].cpu().numpy()
        
        valid_x = valid_featuremaps[fmap_idx].cpu().numpy()
        valid_y = valid_sbrs[:, i].cpu().numpy()

        if type(SVR_PARAM_SPACE["gamma"])==type(str()):
            param_grid = [(c, 'auto') for c in SVR_PARAM_SPACE["C"]]
        else:
            param_grid = list(product(SVR_PARAM_SPACE["C"], SVR_PARAM_SPACE["gamma"]))

        def train_and_evaluate(C, gamma):
            model = SVR(C=C, gamma=gamma, kernel=KERNEL, epsilon=EPSILON)
            model.fit(x, y)
            y_pred_val = model.predict(valid_x)
            mse = mean_squared_error(valid_y, y_pred_val)
            return (C, gamma, mse)

        metrics = Parallel(n_jobs=CORE)(delayed(train_and_evaluate)(C, gamma) for C, gamma in param_grid)
        best_param = min(metrics, key=OBJECTIVE )
        
        best_model = SVR(C=best_param[0], gamma=best_param[1], kernel=KERNEL, epsilon=EPSILON)
        best_model.fit(x, y)

        svrs[fmap_idx].append(best_model)

        search_results = {
            'kernel':KERNEL,
            'epsilon':EPSILON,
            'param_space':SVR_PARAM_SPACE,
            'results':metrics
        }
        joblib.dump(search_results, os.path.join(SAVE_PATH, f"search_results_{fmap_idx}_{SBR_COLUMNS[i]}.joblib"))
joblib.dump(svrs, os.path.join(SAVE_PATH, "best_svrs.joblib"))