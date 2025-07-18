import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from datamodule.DaTdataset import DaTdataset, MinMaxNormalization, SBRMeanStdNormalization, calc_sbr_mean_std
from datamodule.helper import split_data
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

# test

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

acc_running_loss = []
acc_running_acc = []
acc_sensitive = []
acc_specificity = []
acc_auc = []
acc_f1 = []

for n_fold, (train_idx, valid_idx) in enumerate(k_fold.split(train_data, train_labels)):
    
    X_train, _ = [train_data[i] for i in train_idx], [train_data[i] for i in valid_idx]
    y_train, _ = [train_labels[i] for i in train_idx], [train_labels[i] for i in valid_idx]

    train_mean, train_std = calc_sbr_mean_std(X_train)

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

    test_dataset = DaTdataset(test_data, test_labels, transform, sbr_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    model = MODEL(num_classes=1).cuda()
    model.load_state_dict(torch.load(os.path.join(STUDY_NAME, f'fold_{n_fold}', 'best.pth')))

    model.eval()
    running_acc = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, sbrs, labels in tqdm(test_loader, desc="Test", leave=False):
            data, sbrs, labels = data.cuda(), sbrs.cuda(), labels.cuda()
            labels = labels.float()

            outputs, feature_maps = model(data)
            outputs = torch.sigmoid(outputs).squeeze(dim=1)

            predicted = (outputs > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().detach().numpy())


    running_acc = correct / total

    conf_matrix = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds, average='binary')

    print("========================================")
    print(f"{n_fold} fold test result")
    print(f"    Accuracy    = {running_acc:.4f}") 
    print(f"    Sensitivity = {sensitivity:.4f}")
    print(f"    Specificity = {specificity:.4f}")
    print(f"    AUC         = {auc:.4f}")
    print(f"    F1-Score    = {f1:.4f}")
    print(f"    Confusion Matrix:")
    print(f"                CT(Pred)  PD(Pred)")
    print(f"        CT(gt){conf_matrix[0][0]:10}{conf_matrix[0][1]:10}")
    print(f"        PD(gt){conf_matrix[1][0]:10}{conf_matrix[1][1]:10}")

    acc_running_acc.append(running_acc)
    acc_sensitive.append(sensitivity)
    acc_specificity.append(specificity)
    acc_auc.append(auc)
    acc_f1.append(f1)

acc_running_acc = np.array(acc_running_acc)
acc_sensitive = np.array(acc_sensitive)
acc_specificity = np.array(acc_specificity)
acc_auc = np.array(acc_auc)
acc_f1 = np.array(acc_f1)

results = pd.DataFrame({
    "Fold": list(range(0, 5)),
    "Accuracy": acc_running_acc,
    "Sensitivity": acc_sensitive,
    "Specificity": acc_specificity,
    "AUC": acc_auc,
    "F1-Score": acc_f1
})

summary = pd.DataFrame({
    "Metric": ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1-Score"],
    "Mean": [acc_running_acc.mean(), acc_sensitive.mean(), acc_specificity.mean(), acc_auc.mean(), acc_f1.mean()],
    "Std Dev": [acc_running_acc.std(), acc_sensitive.std(), acc_specificity.std(), acc_auc.std(), acc_f1.std()]
})

print("========================================")
print(f"test result")
print(f"    Accuracy    = {acc_running_acc.mean():.4f} (+/- {acc_running_acc.std():.4f})")
print(f"    Sensitivity = {acc_sensitive.mean():.4f} (+/- {acc_sensitive.std():.4f})")
print(f"    Specificity = {acc_specificity.mean():.4f} (+/- {acc_specificity.std():.4f})")
print(f"    AUC         = {acc_auc.mean():.4f} (+/- {acc_auc.std():.4f})")
print(f"    F1-Score    = {acc_f1.mean():.4f} (+/- {acc_f1.std():.4f})")

results.to_excel(os.path.join(STUDY_NAME, "classification_res.xlsx"), index=False)
summary.to_excel(os.path.join(STUDY_NAME, "classification_summary.xlsx"), index=False)