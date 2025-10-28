import numpy as np
import pandas as pd

def label_to_numpy(labels):
    mapping = {
        'Attentive': [1, 0, 0, 0],
        'DrinkingCoffee': [0, 1, 0, 0],
        'UsingMirror': [0, 0, 1, 0],
        'UsingRadio': [0, 0, 0, 1]
    }
    return np.array([mapping[label] for label in labels])

def get_metadata(metadata_path, splits=['train', 'test']):
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata['split'].isin(splits)]
    return metadata

def get_split_data(split_name, metadata, all_data, image_shape):
    subset = metadata[metadata['split'] == split_name]
    idx = subset['index'].values
    labels = subset['class'].values
    data = all_data[idx, :]
    data = data.reshape([-1, *image_shape])
    return data, labels