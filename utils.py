import os
import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_datasets(path: str, n_classes: int, taxonomy: str):
    df_train = pd.read_csv(os.path.join(path, 'train.tsv'), sep="\t", usecols=(1, 2))
    df_validation = pd.read_csv(os.path.join(path, 'dev.tsv'), sep="\t", usecols=(1, 2))
    df_test = pd.read_csv(os.path.join(path, 'test.tsv'), sep="\t", usecols=(1, 2))
    df_train['labels'] = df_train['labels'].apply(correct_labels_in_dataset)
    df_test['labels'] = df_test['labels'].apply(correct_labels_in_dataset)
    df_validation['labels'] = df_validation['labels'].apply(correct_labels_in_dataset)
    if taxonomy == 'ekman':
        df_train['labels'] = df_train['labels'].apply(ekman_labels)
        df_test['labels'] = df_test['labels'].apply(ekman_labels)
        df_validation['labels'] = df_validation['labels'].apply(ekman_labels)
    mlb = MultiLabelBinarizer()
    mlb.fit([[x for x in range(0, n_classes)]])
    df_train['labels'] = df_train['labels'].apply(labels_encode, args=(mlb,))
    df_test['labels'] = df_test['labels'].apply(labels_encode, args=(mlb,))
    df_validation['labels'] = df_validation['labels'].apply(labels_encode, args=(mlb,))
    return df_train, df_validation, df_test


def correct_labels_in_dataset(string):
    # This function transforms each label in the dataset to the correct format
    # example: '0,8,10' = [0, 8, 10]
    aux = list(string.split(","))
    for i, item in enumerate(aux):
        aux[i] = int(item)
    return aux


def ekman_labels(vector):
    with open('emotion_dict.json', 'r', encoding='utf-8') as emotion_dict:
        emotion_dict = json.loads(emotion_dict.read())
    with open('ekman_mapping.json', 'r', encoding='utf-8') as ekman_mapping:
        ekman_mapping = json.loads(ekman_mapping.read())

    new_vector = list()
    for label in vector:
        label = list(emotion_dict.keys())[list(emotion_dict.values()).index(label)]
        for key, value in ekman_mapping.items():
            if label in value:
                label = key
                break
        label = list(ekman_mapping.keys()).index(label)
        new_vector.append(label)
    return new_vector


def labels_encode(vector, mlb: MultiLabelBinarizer):
    label = mlb.transform([vector])
    label = list(label[0])
    return label


def labels_decode(vector, mlb: MultiLabelBinarizer):
    label = np.array(vector)
    label = np.reshape(label, (1, label.shape[0]))
    return list(mlb.inverse_transform(label)[0])


def class_balanced_loss(n_classes, samples_per_classes, b_labels, beta, no_cuda):
    effective_num = 1.0 - np.power(beta, samples_per_classes)
    weights_for_samples = (1.0 - beta) / np.array(effective_num)
    weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * n_classes
    b_labels = b_labels.to('cpu')
    weights_for_samples = torch.tensor(weights_for_samples).float()
    weights_for_samples = weights_for_samples.unsqueeze(0)
    weights_for_samples = torch.tensor(np.array(weights_for_samples.repeat(b_labels.shape[0], 1) * b_labels))
    weights_for_samples = weights_for_samples.sum(1)
    weights_for_samples = weights_for_samples.unsqueeze(1)
    weights_for_samples = weights_for_samples.repeat(1, n_classes)
    if torch.cuda.is_available() and not no_cuda:
        weights_for_samples = weights_for_samples.to(torch.device("cuda"))
    return weights_for_samples

