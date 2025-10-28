from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation
from tensorflow.keras import optimizers

def dense_classifier(hidden_layers, nn_params, dropout=0.5):
    model = Sequential()
    model.add(Flatten(input_shape=nn_params['input_shape']))
    for units in hidden_layers:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nn_params['output_neurons'], activation=nn_params['output_activation']))
    model.compile(loss=nn_params['loss'],
                  optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.95),
                  metrics=['accuracy'])
    return model

def cnn_classifier(num_layers, nn_params, dropout=0.5):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=nn_params['input_shape']))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    for _ in range(num_layers - 1):
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nn_params['output_neurons'], activation=nn_params['output_activation']))
    model.compile(loss=nn_params['loss'], optimizer='rmsprop', metrics=['accuracy'])
    return model