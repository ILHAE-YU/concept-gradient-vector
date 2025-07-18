from models.DeepPDNet import DeepPDNet
from models.ResNet import ResNet
import torch
import torch.nn as nn
import numpy as np

# Data
RANDOM_SEED = 42
TEST_FIX_SEED = 42
SPLIT_RATIO = {
    "train":60,
    "prototype":20,
    "test":20
}
EXCLUDE = True
TABLE_PATH = "./data//DaT_preprocessed.csv"
DATA_PATH = "./data/DaT_preprocessed"
SBR_COLUMNS = [
    "DATSCAN_CAUDATE_R",
    "DATSCAN_CAUDATE_L",
    "DATSCAN_PUTAMEN_R",
    "DATSCAN_PUTAMEN_L",
    "DATSCAN_PUTAMEN_R_ANT",
    "DATSCAN_PUTAMEN_L_ANT",
]


# Classification
MODEL = DeepPDNet
EPOCH = 20
OPTIMIZER = torch.optim.SGD
OTIM_PARAMS = {
    "lr":0.001,
    "momentum":0.9, 
    "weight_decay":0.0001
}
CRITERION = nn.BCELoss
BATCH = 64


# SBR regression
SVR_PARAM_SPACE = {
    "C": np.array([1.0]),
    "gamma": np.array([1/512]),
}
CORE = -4
OBJECTIVE = lambda x : x[2]
KERNEL = "rbf"
EPSILON = 0.01


# CGV 
TEST_FOLD = 0
CONCEPT_UNIT = -1.0


# results
STUDY_NAME = "./train-PPMI-DeepPDNet"