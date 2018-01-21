import time

import numpy as np

import keras
import keras.layers as layers
from keras.optimizers import RMSprop
import keras.applications.vgg19 as vg
from keras.models import Model, Sequential

import tensorflow as tf

np.random.seed(0)

def main():

    ## NEW ENTRY ##

    # x = np.load('input_transformed_scans.npz')
    # y = np.load('labels.npz')
    size = 675

    # Initialize cnn
    cnn = vg.VGG19(include_top=True, weights=None, pooling='None', classes=2)
    lstm = Model(cnn.get_layer(index=18).output,
                 layers.LSTM(size,
                             activation='sigmoid',
                             kernel_regularizer=keras.regularizers.l2(0.)))

    # Build sequence
    fragility_model = Sequential()
    fragility_model.add(layers.Bidirectional(
        lstm,
        merge_mode='mul'
    ))
    fragility_model.add(layers.Dense(1, activation='sigmoid'))

    # Compile
    rmsprop = RMSprop(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model = Model()
    model.compile(optimizer=rmsprop,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train
    model.fit(x=x, y=y,
              batch_size=128,
              epochs=1000,
              verbose=2,
              validation_split=0.4)


    print('Done!')


if __name__ == "__main__":
    main()
