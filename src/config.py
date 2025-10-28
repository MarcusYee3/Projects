IMAGE_SHAPE = (64, 64, 3)
NN_PARAMS = {
    'input_shape': IMAGE_SHAPE,
    'output_neurons': 4,
    'loss': 'categorical_crossentropy',
    'output_activation': 'softmax'
}
DATA_PATHS = {
    'metadata': './data/raw/metadata.csv',
    'images': './data/raw/image_data.npy'
}