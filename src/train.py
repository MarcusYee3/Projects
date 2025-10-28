import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from config import NN_PARAMS, IMAGE_SHAPE, DATA_PATHS
from data_utils import get_metadata, get_split_data, label_to_numpy
from model_utils import cnn_classifier
from visualize import plot_saliency

all_data = np.load(DATA_PATHS['images'])
metadata = get_metadata(DATA_PATHS['metadata'], ['train', 'test'])

X_train, y_train_str = get_split_data('train', metadata, all_data, IMAGE_SHAPE)
X_test, y_test_str = get_split_data('test', metadata, all_data, IMAGE_SHAPE)

y_train = label_to_numpy(y_train_str)
y_test = label_to_numpy(y_test_str)

cnn = cnn_classifier(num_layers=5, nn_params=NN_PARAMS)
checkpoint = ModelCheckpoint('./models/best_model.h5', save_best_only=True, monitor='val_accuracy')

cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[checkpoint])

# Visualize saliency on sample data
plot_saliency(cnn, X_test[:4], y_test_str[:4])