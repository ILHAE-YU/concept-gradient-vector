import os
import copy
from tqdm import tqdm
import datetime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

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

# training
if STUDY_NAME == None:
    STUDY_NAME = "train" + "-" + datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S")
SAVE_DIR = STUDY_NAME

os.makedirs(SAVE_DIR)

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for n_fold, (train_idx, valid_idx) in enumerate(k_fold.split(train_data, train_labels)):

    FOLD_SAVE_PATH = os.path.join(SAVE_DIR, f'fold_{n_fold}')
    os.makedirs(FOLD_SAVE_PATH)
    os.makedirs(os.path.join(FOLD_SAVE_PATH, 'weight'))

    X_train, X_valid = [train_data[i] for i in train_idx], [train_data[i] for i in valid_idx]
    y_train, y_valid = [train_labels[i] for i in train_idx], [train_labels[i] for i in valid_idx]

    train_mean, train_std = calc_sbr_mean_std(X_train)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.type(torch.float32)),
        MinMaxNormalization(0.0, 1.0),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Lambda(lambda x: x[..., 13:-14, 13:-14, 6:-7]),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.type(torch.float32)),
        MinMaxNormalization(0.0, 1.0),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
        transforms.Lambda(lambda x: x[..., 13:-14, 13:-14, 6:-7]),

    ])
    sbr_transform = transforms.Compose([
        SBRMeanStdNormalization(train_mean, train_std)
    ])

    train_dataset = DaTdataset(X_train, y_train, train_transform, sbr_transform)
    valid_dataset = DaTdataset(X_valid, y_valid, valid_transform, sbr_transform)
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False)

    model = MODEL(num_classes=1).cuda()
    criterion = CRITERION()
    optimizer = OPTIMIZER(model.parameters(), **OTIM_PARAMS)
    best_model_state = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    writer = SummaryWriter(log_dir=os.path.join(FOLD_SAVE_PATH))
    
    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_preds = []
        all_probs = []

        for data, sbrs, labels in tqdm(train_loader, desc=f"Epoch: {epoch}, train", leave=False):
            data, sbrs, labels = data.cuda(), sbrs.cuda(), labels.cuda()
            labels = labels.float()

            optimizer.zero_grad()
            outputs, feature_maps = model(data)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().detach().numpy())

        running_acc = correct / total
        running_loss = running_loss / len(train_loader)

        conf_matrix = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = conf_matrix.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, average='binary')

        writer.add_scalar('Loss/train', running_loss, epoch)
        writer.add_scalar('Accuracy/train', running_acc, epoch)
        writer.add_scalar('Sensitivity/train', sensitivity, epoch)
        writer.add_scalar('Specificity/train', specificity, epoch)
        writer.add_scalar('AUC/train', auc, epoch)
        writer.add_scalar('F1-Score/train', f1, epoch)

        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, sbrs, labels in tqdm(valid_loader, desc=f"Epoch: {epoch}, valid", leave=False):
                data, sbrs, labels = data.cuda(), sbrs.cuda(), labels.cuda()
                labels = labels.float()

                outputs, feature_maps = model(data)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze(dim=1)

                loss_cls = criterion(outputs, labels)
                loss = loss_cls

                running_loss += loss.item()
                predicted = (outputs > 0.5).long()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(outputs.cpu().detach().numpy())

        running_acc = correct / total
        running_loss = running_loss / len(valid_loader)

        conf_matrix = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = conf_matrix.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, average='binary')

        writer.add_scalar('Loss/valid', running_loss, epoch)
        writer.add_scalar('Accuracy/valid', running_acc, epoch)
        writer.add_scalar('Sensitivity/valid', sensitivity, epoch)
        writer.add_scalar('Specificity/valid', specificity, epoch)
        writer.add_scalar('AUC/valid', auc, epoch)
        writer.add_scalar('F1-Score/valid', f1, epoch)

        if running_loss < best_loss:
            best_loss = running_loss
            best_model_state = copy.deepcopy(model.state_dict())

        torch.save(copy.deepcopy(model.state_dict()), os.path.join(FOLD_SAVE_PATH, 'weight', f'{epoch}.pth'))
    torch.save(best_model_state, os.path.join(FOLD_SAVE_PATH, f'best.pth'))