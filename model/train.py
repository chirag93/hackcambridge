import time

import numpy as np
from functools import reduce
import math as m

######### import DNN frameworks #########
import tensorflow as tf
import keras

######### import DNN for training using GPUs #########
from keras.utils.training_utils import multi_gpu_model

# utility functionality for keras - preprocessing sequential data
from keras.preprocessing import sequence 
# preprocessing - image data
from keras.preprocessing.image import ImageDataGenerator


np.random.seed(1234)

def train(model, images, labels, fold,
    batch_size=32, num_epochs=5):
    """
    A sample training function which loops over the training set and evaluates the network
    on the validation set after each epoch. Evaluates the network on the training set
    whenever the
    :param images: input images
    :param labels: target labels
    :param fold: tuple of (train, test) index numbers
    :param model_type: model type ('cnn', '1dconv', 'maxpool', 'lstm', 'mix')
    :param batch_size: batch size for training
    :param num_epochs: number of epochs of dataset to go over for training
    :return: none
    """
    num_classes = len(np.unique(labels))
    
    # get train set and test set using utility reformatting function
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(images, labels, fold)
    X_train = X_train.astype("float32", casting='unsafe')
    X_val = X_val.astype("float32", casting='unsafe')
    X_test = X_test.astype("float32", casting='unsafe')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
 
    # ADAM
    ADAM = keras.optimizers.Adam(lr=0.001, 
        beta_1=0.9, beta_2=0.999, 
        epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', 
        optimizer=ADAM, 
        metrics=["accuracy"])

    # the set of callbacks
    aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True,
        fill_mode="nearest")
    callbacks = [LearningRateScheduler(poly_decay)]
    INIT_LR = 5e-3

    HH = model.fit_generator(
        aug.flow(X_train, y_train, batch_size=64 * G), # adds augmentation to data using generator
        validation_data=(X_test, y_test),  
        steps_per_epoch=len(X_train) // (64 * G),    #
        epochs=NUM_EPOCHS,
        callbacks=callbacks, verbose=2)

    return HH

def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0
    # compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    # return the new learning rate
    return alpha

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    from utils import reformatInput

    # :param input_vars: list of EEG images (one image per time window)
    # :param nb_classes: number of classes
    # :param imsize: size of the input image (assumes a square input)
    # :param n_colors: number of color channels in the image
    # :param n_timewin: number of time windows in the snippet

    # test this function using MINST
    input_vars = []
    nb_classes = 10 # for MINS
    # imsize = 
    # build_convpool_max(input_vars, nb_classes, imsize=32, n_colors=3, n_timewin=3)


    # Load electrode locations
    print('Loading data...')
    locs = scipy.io.loadmat('../Sample data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    feats = scipy.io.loadmat('../Sample data/FeatureMat_timeWin.mat')['features']
    subj_nums = np.squeeze(scipy.io.loadmat('../Sample data/trials_subNums.mat')['subjectNum'])
    # Leave-Subject-Out cross validation
    fold_pairs = []
    for i in np.unique(subj_nums):
        ts = subj_nums == i
        tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
        ts = np.squeeze(np.nonzero(ts))
        np.random.shuffle(tr)  # Shuffle indices
        np.random.shuffle(ts)
        fold_pairs.append((tr, ts))

    # CNN Mode
    print('Generating images...')
    # Find the average response over time windows
    av_feats = reduce(lambda x, y: x+y, [feats[:, i*192:(i+1)*192] for i in range(feats.shape[1] / 192)])
    av_feats = av_feats / (feats.shape[1] / 192)
    images = gen_images(np.array(locs_2d),
                                  av_feats,
                                  32, normalize=False)
    print('\n')

    # Class labels should start from 0
    print('Training the CNN Model...')
    train(images, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'cnn')

    # Conv-LSTM Mode
    print('Generating images for all time windows...')
    images_timewin = np.array([gen_images(np.array(locs_2d),
                                                    feats[:, i * 192:(i + 1) * 192], 32, normalize=False) for i in
                                         range(feats.shape[1] / 192)
                                         ])
    print('\n')
    print('Training the LSTM-CONV Model...')
    train(images_timewin, np.squeeze(feats[:, -1]) - 1, fold_pairs[2], 'mix')

    print('Done!')
