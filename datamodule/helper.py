def split_data(table_path, data_path, sbr_columns, split_ratio, exclude=True, random_seed=42, test_fix_seed=42):
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    table = pd.read_csv(table_path, index_col=0)
    pd_table = table[table['cohort'] == 'PD']
    control_table = table[table['cohort'] == 'Control']

    pd_n = len(pd_table)
    control_n = len(control_table)

    if exclude == True:
        excluding_pd_n = pd_n - control_n
        excluding_pd_table = pd_table.sample(n=excluding_pd_n, random_state=random_seed)
        pd_table = pd_table.drop(excluding_pd_table.index)

    sbr_datas = pd_table[sbr_columns].values.tolist() + control_table[sbr_columns].values.tolist()
    sbr_datas_ex = excluding_pd_table[sbr_columns].values.tolist()

    labels = pd_table.cohort.to_list() + control_table.cohort.to_list()
    excluded_labels = excluding_pd_table.cohort.to_list()

    data = pd_table.subject_number.to_list() + control_table.subject_number.to_list()
    data = [[os.path.join(data_path, str(x), 'DaT.nii')] + sbr_datas[i] for i, x in enumerate(data)]

    excluded_data = excluding_pd_table.subject_number.to_list()
    excluded_data = [[os.path.join(data_path, str(x), 'DaT.nii')] + sbr_datas_ex[i] for i, x in enumerate(excluded_data)]

    temp_ratio = (split_ratio['train'] + split_ratio['prototype']) / (split_ratio['train'] + split_ratio['prototype'] + split_ratio['test'])
    test_data, temp_data, test_labels, temp_labels = train_test_split(data, labels, test_size=temp_ratio, stratify=labels, random_state=test_fix_seed)

    train_ratio = (split_ratio['train']) / (split_ratio['train'] + split_ratio['prototype'])
    prototype_data, train_data, prototype_labels, train_labels = train_test_split(temp_data, temp_labels, test_size=train_ratio, stratify=temp_labels, random_state=random_seed)

    return train_data, train_labels, prototype_data, prototype_labels, test_data, test_labels, excluded_data, excluded_labels


def feature_extract(model, prototype_loader, valid_loader, test_loader):
    import os
    import torch
    from tqdm import tqdm
    
    prototype_featuremaps = None
    prototype_sbrs = None

    with torch.no_grad():
        for data, sbrs, labels in tqdm(prototype_loader, desc="Feature Ext", leave=False):
            data, sbrs, labels = data.cuda(), sbrs.cuda(), labels.cuda()
            _, feature_maps = model(data)
            if prototype_featuremaps == None and prototype_sbrs == None:
                prototype_featuremaps = [feature_maps[i].flatten(start_dim=1) for i in range(len(feature_maps))]
                prototype_sbrs = sbrs
            else:
                for i in range(len(feature_maps)):
                    prototype_featuremaps[i] = torch.cat((prototype_featuremaps[i], feature_maps[i].flatten(start_dim=1)), dim=0)
                prototype_sbrs = torch.cat((prototype_sbrs, sbrs), dim=0)

    valid_featuremaps = None
    valid_sbrs = None

    with torch.no_grad():
        for data, sbrs, labels in tqdm(valid_loader, desc="Feature Ext", leave=False):
            data, sbrs, labels = data.cuda(), sbrs.cuda(), labels.cuda()
            _, feature_maps = model(data)
            if valid_featuremaps == None and valid_sbrs == None:
                valid_featuremaps = [feature_maps[i].flatten(start_dim=1) for i in range(len(feature_maps))]
                valid_sbrs = sbrs
            else:
                for i in range(len(feature_maps)):
                    valid_featuremaps[i] = torch.cat((valid_featuremaps[i], feature_maps[i].flatten(start_dim=1)), dim=0)
                valid_sbrs = torch.cat((valid_sbrs, sbrs), dim=0)

    test_featuremaps = None
    test_sbrs = None

    with torch.no_grad():
        for data, sbrs, labels in tqdm(test_loader, desc="Feature Ext", leave=False):
            data, sbrs, labels = data.cuda(), sbrs.cuda(), labels.cuda()
            _, feature_maps = model(data)

            if test_featuremaps == None and test_sbrs == None:
                test_featuremaps = [feature_maps[i].flatten(start_dim=1) for i in range(len(feature_maps))]
                test_sbrs = sbrs
            else:
                for i in range(len(feature_maps)):
                    test_featuremaps[i] = torch.cat((test_featuremaps[i], feature_maps[i].flatten(start_dim=1)), dim=0)
                test_sbrs = torch.cat((test_sbrs, sbrs), dim=0)
                
    feature_maps_length = len(prototype_featuremaps)

    for fmaps, sbrs, tttype in [[prototype_featuremaps, prototype_sbrs, "prototype"],
                            [valid_featuremaps, valid_sbrs, "valid"],
                            [test_featuremaps, test_sbrs, "test"]]:
        print(tttype)
        for fmap in fmaps:
            print("    fmap:", fmap.shape, "", sbrs.shape)

    return prototype_featuremaps, prototype_sbrs, valid_featuremaps, valid_sbrs, test_featuremaps, test_sbrs, feature_maps_length