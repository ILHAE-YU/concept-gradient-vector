# import library 
import os
import sys
from tqdm import tqdm
import pandas as pd
import joblib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error

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

SAVE_PATH = os.path.join(STUDY_NAME, f"fold_{TEST_FOLD}", "cgv")

svrs = joblib.load(os.path.join(SAVE_PATH, "best_svrs.joblib"))

SAVE_PATH = os.path.join(SAVE_PATH, 'regression_results')
os.makedirs(SAVE_PATH, exist_ok=True)

for fmap_idx in range(feature_maps_length):

    r2s =[]
    mses = []
    pearsons = []

    for i in tqdm(range(len(SBR_COLUMNS)), desc=f"{fmap_idx} svrs test", leave=False):    
        x = test_featuremaps[fmap_idx].cpu().numpy()
        y_reg = test_sbrs[:, i].cpu().numpy()
        y_cls = test_loader.dataset.labels.cpu().numpy()

        model = svrs[fmap_idx][i]
        pred_y = model.predict(x)

        r2 = r2_score(y_reg, pred_y)
        mse = mean_squared_error(y_reg, pred_y)
        pearson_corr, _ = pearsonr(y_reg, pred_y)

        r2s.append(r2)
        mses.append(mse)
        pearsons.append(pearson_corr)

        plt.figure(figsize=(5,6))
        scatter = plt.scatter(y_reg, pred_y, c=y_cls, cmap="viridis", alpha=0.5, s=16)

        plt.plot([y_reg.min(), y_reg.max()], 
                [y_reg.min(), y_reg.max()], 
                'r--', label="Ideal", linewidth=2)

        # Add labels and title
        plt.xlabel("True Normalized SBR")
        plt.ylabel("Predicted Normalized SBR")
        plt.title(f"{fmap_idx} - {SBR_COLUMNS[i]}\n"
                  + f"R-sq: {r2:.4}\n"
                  + f"MSE : {mse:.4}\n"
                  + f"Cor.: {pearson_corr:.4}\n"
                  + f"C : {model.C}, G: {model._gamma}\n"
                  + f"SV : {model.support_vectors_.shape[0]}")
        plt.grid(True)
        plt.tight_layout()
        output_path = os.path.join(SAVE_PATH, f"{fmap_idx}_{i}_{SBR_COLUMNS[i]}.png")
        plt.savefig(output_path)
        plt.close()

    results = pd.DataFrame({
        "SBR":SBR_COLUMNS,
        "r2":r2s,
        "mse":mses,
        "cor":pearsons
    })

    results.to_excel(os.path.join(SAVE_PATH, f"regression_res_{fmap_idx}.xlsx"), index=False)
