import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import nibabel as nib
from sklearn.preprocessing import LabelEncoder

# pd 1
# control 0

class DaTdataset(Dataset):

    def __init__(self, data, labels, transform=None, sbr_transform=None, train=True, label_encode=True):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.sbr_transform = sbr_transform
        
        if label_encode==True:
            label_encoder = LabelEncoder()
            self.labels = label_encoder.fit_transform(self.labels)
        self.labels = torch.tensor(self.labels).to(torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]
        img_path = self.data[idx][0]
        image = nib.load(img_path).get_fdata()
        sbrs = self.data[idx][1:]
        sbrs = torch.tensor(sbrs)
        
        if self.transform:
            image = self.transform(image)

        if self.sbr_transform:
            sbrs = self.sbr_transform(sbrs)

        return image, sbrs, label

class MeanStdNormalization:
    def __init__(self, mean_value=0.0, std_value=1.0):
        self.mean = mean_value
        self.std = std_value
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

class SBRMeanStdNormalization:
    def __init__(self, mean_value=None, std_value=None):
        self.mean = mean_value if mean_value is not None else torch.zeros(102)
        self.std = std_value if std_value is not None else torch.ones(102)
        pass
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class MinMaxNormalization:
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value
    
    def __call__(self, tensor):
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        
        scaled_tensor = normalized_tensor * (self.max_value - self.min_value) + self.min_value
        return scaled_tensor
    

if __name__ == "__main__":
    table = pd.read_csv("DaT_baseline_sub.csv", index_col=0)
    table = table[table['cohort'].isin(['PD', 'Control'])]

    base_dir = "DaT_baseline_nii_align_sort"

    data = [os.path.join(base_dir, str(row['subject_number']), "DaT.nii") for _, row in table.iterrows()]
    labels = table['cohort'].tolist()

    dataset = DaTdataset(data, labels)
    for i in range(len(dataset)):
        print(dataset[i][0].shape)

def calc_sbr_mean_std(data):
    sbr_len = len(data[0])-1
    feature_sum = torch.zeros(sbr_len)
    feature_squared_sum = torch.zeros(sbr_len)
    num_samples = len(data)

    for dt in data:
        features = torch.tensor(dt[1:])
        feature_sum += features
        feature_squared_sum += features ** 2

    feature_mean = feature_sum / num_samples
    feature_std = torch.sqrt((feature_squared_sum / num_samples) - (feature_mean ** 2))

    return feature_mean, feature_std