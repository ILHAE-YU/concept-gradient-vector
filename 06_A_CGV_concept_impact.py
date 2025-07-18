# import library 
import os
from tqdm import tqdm
import numpy as np
import joblib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

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

SVR_PATH = os.path.join(STUDY_NAME, f"fold_{TEST_FOLD}", "cgv")
svrs = joblib.load(os.path.join(SVR_PATH, "best_svrs.joblib"))

model = MODEL(num_classes=1).cuda()
WEIGHT_DIR = os.path.join(STUDY_NAME, f"fold_{TEST_FOLD}", 'best.pth')
model.load_state_dict(torch.load(WEIGHT_DIR))
model.eval()

# f; feature map 
# n; sample
# c; concept
concept_saliency_map_arr = [] # (n, f, c, w, h, d)
datas = [] # (n, w, h, d)
true_sbr_arr = [] # (n, c)
impact_arr = [] # (n, f, c)
prob_perturbated = [] # (n, f, c)
labels = [] # (n)
prob = [] # (n)
predicted_sbr_arr = [] # (n, f, c)
original_saliency_map_arr = [] # (n, w, h, d)


h_norm = [] # (n, f, c)
c_norm = []

for data, sbrs, label in tqdm(test_loader, desc="compute impact", leave=False):

    data = data.cuda()
    data.requires_grad = True
    outputs, feature_maps = model(data)

    prob.append(torch.sigmoid(outputs[0]).item())
    labels.append(label.item())
    datas.append(data.clone().detach().squeeze().cpu().numpy())
    true_sbr_arr.append(sbrs.squeeze().numpy())

    for feature_map in feature_maps:
        feature_map.retain_grad()

    target_class_output = outputs[0]
    target_class_output = torch.sigmoid(target_class_output).squeeze()

    model.zero_grad()
    target_class_output.backward(retain_graph=True)

    original_saliency_map_arr.append(data.grad.clone().detach().cpu().squeeze().numpy())

    feature_maps_gradient = []
    for feature_map in feature_maps:
        feature_maps_gradient.append(feature_map.grad.clone())

    temp_impact_d1 = []
    temp_concept_saliency_map_d1 = []
    temp_predicted_sbr_d1 = []
    prob_perturbated_d1 = []

    h_norm_d1 = []
    c_norm_d1 = []

    for fmap_idx, gradient in enumerate(feature_maps_gradient):

        temp_impact_d0 = []
        temp_concept_saliency_map_d0 = []
        temp_predicted_sbr_d0 = []
        prob_perturbated_d0 = []

        h_norm_d0 = []
        c_norm_d0 = []


        for concept_idx, svr in enumerate(svrs[fmap_idx]):

            ### SVR gradient
            support_vectors = svr.support_vectors_
            dual_coef = svr.dual_coef_
            gamma = svr._gamma
            intecept = svr._intercept_

            
            shape = feature_maps[fmap_idx].shape
            h = feature_maps[fmap_idx].clone().flatten().detach().cpu().numpy()

            K = np.exp(-gamma * np.linalg.norm( h - support_vectors, axis=1) ** 2 )
            grad_K = -2 * gamma * ( h - support_vectors ) * K[:, None]

            svr_grad = np.sum( np.squeeze(dual_coef)[:,None] * grad_K, axis=0)
            svr_grad_norm = np.linalg.norm(svr_grad)
            unit_svr_grad = (svr_grad) / (svr_grad_norm) ** 2

            ### impact 
            h_perturbated = h + unit_svr_grad * CONCEPT_UNIT
            h_perturbated_reshape = np.reshape(h_perturbated, shape)

            output_perturbated = model._forward(torch.Tensor(h_perturbated_reshape).cuda())
            output_prob = torch.sigmoid(outputs[0])
            output_perturbated_prob = torch.sigmoid(output_perturbated)
            
            impact = output_perturbated_prob - output_prob

            ### Concept Saliency
            model.zero_grad()
            data.grad.zero_()
            
            feature_maps[fmap_idx].backward(
                gradient=torch.tensor(
                    # unit_svr_grad.reshape(shape), 
                    svr_grad.reshape(shape), 
                    dtype=torch.float32, 
                    device="cuda"
                ),
                retain_graph=True
            )

            h_norm_d0.append(np.linalg.norm(h))
            c_norm_d0.append(svr_grad_norm)

            temp_impact_d0.append(impact.clone().detach().cpu().item())
            temp_concept_saliency_map_d0.append(data.grad.clone().detach().cpu().squeeze().numpy())
            temp_predicted_sbr_d0.append(svr.predict(h.reshape(1,-1)))
            prob_perturbated_d0.append(output_perturbated_prob.item())
            
        temp_impact_d1.append(temp_impact_d0)
        temp_concept_saliency_map_d1.append(temp_concept_saliency_map_d0)
        temp_predicted_sbr_d1.append(temp_predicted_sbr_d0)
        prob_perturbated_d1.append(prob_perturbated_d0)
    
        h_norm_d1.append(h_norm_d0)
        c_norm_d1.append(c_norm_d0)

    impact_arr.append(temp_impact_d1)
    concept_saliency_map_arr.append(temp_concept_saliency_map_d1)
    predicted_sbr_arr.append(temp_predicted_sbr_d1)
    prob_perturbated.append(prob_perturbated_d1)

    h_norm.append(h_norm_d1)
    c_norm.append(c_norm_d1)

impact_arr = np.array(impact_arr)
concept_saliency_map_arr = np.array(concept_saliency_map_arr)
datas = np.array(datas)
labels = np.array(labels)
prob = np.array(prob)
true_sbr_arr = np.array(true_sbr_arr)
predicted_sbr_arr = np.array(predicted_sbr_arr)
prob_perturbated = np.array(prob_perturbated)
original_saliency_map_arr = np.array(original_saliency_map_arr)

h_norm = np.array(h_norm)
c_norm = np.array(c_norm)


results = {
    "impact": impact_arr,
    "concept_saliency_map": concept_saliency_map_arr,
    "data": datas,
    "label": labels,
    "prob": prob,
    "sbr": true_sbr_arr,
    "sbr_pred": predicted_sbr_arr,
    "prob_perturbated": prob_perturbated,
    "original_saliency_map": original_saliency_map_arr
}

joblib.dump(results, os.path.join(SVR_PATH, "concept_impact_results.joblib"))
