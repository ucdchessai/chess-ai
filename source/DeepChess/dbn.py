import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import time
import numpy as np
from keras.models import Model, Sequential, clone_model
from keras.layers import Input, Dense, Activation, Concatenate
from keras.utils import Sequence
from keras import backend as K

### GLOBALS AND HYPER PARAMETER SETTINGS ###
dnb_epochs = 200
deep_chess_epochs = 1000

# Setup the autoencoder, a.k.a the feature extractor

""" Batch size, Layer Sizes and sample size as proposed in paper """
batch_size = 512
autoencoderLayers = [773, 600, 400, 200, 100]
deepChessLayers = [400, 200, 100, 2]
dataSetSize = 1000000
sampleSize = 800000

# ------------> Change these depending on the dataset <------------
whiteWonFile = "../data/whiteWin.npy"
whiteLostFile = "../data/blackWin.npy"

""" Batch size, Layer Sizes and sample size tayloerd to our encoding """
# NOTE: sampleSize >= batch_size. Otherwise it will throw an error.
# batch_size = 256
# autoencoderLayers = [69, 69, 69, 60, 40]
# deepChessLayers = [60, 40, 20, 2]
# dataSetSize = 1000000
# sampleSize = 600000
# whiteWonFile = "./data/white69.npy"
# whiteLostFile = "./data/black69.npy"

dbnLayers = len(autoencoderLayers) - 1

numWhiteWon = dataSetSize
numWhiteLost = dataSetSize

# FILE LOADING

mat = np.zeros((numWhiteWon+numWhiteLost, autoencoderLayers[0]))
mat[:numWhiteWon] = np.load(whiteWonFile)[:dataSetSize]
mat[numWhiteLost:] = np.load(whiteLostFile)[:dataSetSize]
# data_mat = mat.copy()
np.random.shuffle(mat)

# --------- SETUP DEEP BELIEF NETOWORK ----------- #

""" 
Here we train the layers of the autoencoder one by one - that is,
if the first hidden layer has 600 nodes, then we train it as:
    InputLayerNodes:600:InputLayerNodes
and if the second hidden layer has 400 nodes, then:
    600:400:600
And so on, and so forth. 
This means our input and output layers are the same, and the
significance of this is that ???
We save the the weights between two
successive layers in the variable `weightMatrix`

MORE ON AUTOENCODERS: https://en.wikipedia.org/wiki/Autoencoder
"""

weightMatrix = []
shape_vec = []

for i in range(dbnLayers):
    """ INITIALIZE A SEQUENTIAL MODEL """
    dbn_model = Sequential()
    dbn_model.add(Dense(autoencoderLayers[i+1], activation='relu',
                        input_dim=autoencoderLayers[i]))
    dbn_model.add(Dense(autoencoderLayers[i], activation='relu'))
    dbn_model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])

    dbn_model.summary()

    """ TRAIN MODEL """
    dbn_model.fit(mat, mat, batch_size=batch_size, epochs=dnb_epochs)
    # score = dbn_model.evaluate(mat, mat, batch_size=batch_size)
    # print(score)

    # GET THE WEIGHT MATRIX
    weightMatrix.append(dbn_model.layers[0].get_weights())
    # Get the outputs of the hidden layer
    getHiddenOuptut = K.function(
        [dbn_model.input], [dbn_model.layers[0].output])
    mat = getHiddenOuptut([mat])[0]
    # print("HIDDEN SHAPE: ", getHiddenOuptut)
    # print(weightMatrix)
    # shape_vec.append(mat.shape)


# print("WEIGHT MATRIX SHAPE:", len(
#     weightMatrix[0][0]), len(weightMatrix[0][0][0]))
# print("WEIGHT MATRIX SHAPE:", len(
#     weightMatrix[1][0]), len(weightMatrix[1][0][0]))
# print("WEIGHT MATRIX SHAPE:", len(
#     weightMatrix[2][0]), len(weightMatrix[2][0][0]))
# print("WEIGHT MATRIX SHAPE:", len(
#     weightMatrix[3][0]), len(weightMatrix[3][0][0]))


""" Now that we have trained the autoecoding layers, let us 
    construct the actual DBN dbn_model, and make them 'Siamese network'
"""
dbn_model = [None]*2
for i in range(2):
    dbn_model[i] = Sequential()
    dbn_model[i].add(Dense(autoencoderLayers[1], activation='relu',
                           input_dim=autoencoderLayers[0], trainable=False))
    dbn_model[i].add(
        Dense(autoencoderLayers[2], activation='relu', trainable=False))
    dbn_model[i].add(
        Dense(autoencoderLayers[3], activation='relu', trainable=False))
    dbn_model[i].add(
        Dense(autoencoderLayers[4], activation='relu', trainable=False))
    dbn_model[i].compile(optimizer='adam',
                         loss='mse',
                         metrics=['accuracy'])
    dbn_model[i].summary()

    dbn_model[i].layers[0].set_weights(weightMatrix[0])
    dbn_model[i].layers[1].set_weights(weightMatrix[1])
    dbn_model[i].layers[2].set_weights(weightMatrix[2])
    dbn_model[i].layers[3].set_weights(weightMatrix[3])

# SAVE THE INTERMEDIATE MODEL IN CASE OF CRASH
timestr = time.strftime("%Y%m%d-%H%M%S")
model_filename = "dbn-model.h5"
dbn_model[0].save(os.path.join("./models/", model_filename))

