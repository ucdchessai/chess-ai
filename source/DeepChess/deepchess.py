import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from keras import backend as K
from keras.utils import Sequence
from keras.layers import Input, Dense, Activation, Concatenate
from keras.models import Model, Sequential, clone_model
import numpy as np
import time
import os

### GLOBALS AND HYPER PARAMETER SETTINGS ###
dnb_epochs = 200
deep_chess_epochs = 1000

# Setup the autoencoder, a.k.a the feature extractor

""" Batch size, Layer Sizes and sample size as proposed in paper """
# batch_size = 512
# autoencoderLayers = [773, 600, 400, 200, 100]
# deepChessLayers = [400, 200, 100, 2]
# dataSetSize = 1000000
# sampleSize = 1000000
# whiteWonFile = "./data/whiteBit.npy"
# whiteLostFile = "./data/blackBit.npy"

""" Batch size, Layer Sizes and sample size tayloerd to our encoding """
batch_size = 256
autoencoderLayers = [64, 64, 64, 60, 60]
deepChessLayers = [60, 40, 20, 2]
dataSetSize = 1000000
sampleSize = 700000
whiteWonFile = "./data/white.npy"
whiteLostFile = "./data/black.npy"

dbnLayers = len(autoencoderLayers) - 1

numWhiteWon = dataSetSize
numWhiteLost = dataSetSize

# FILE LOADING

mat = np.zeros((numWhiteWon+numWhiteLost, autoencoderLayers[0]))
mat[:numWhiteWon] = np.load(whiteWonFile)[:dataSetSize]
mat[numWhiteLost:] = np.load(whiteLostFile)[:dataSetSize]
data_mat = mat.copy()
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
model_filename = "dbn-" + timestr + ".h5"
dbn_model[0].save(os.path.join("./models/", model_filename))

#----------- BEGIN DEEP CHESS IMPLEMENTATION -------------#


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, batch_size=32, sampleSize=100):
        'Initialization'
        self.batch_size = batch_size
        self.sampleSize = sampleSize

        global data_mat
        global numWhiteWon
        self.whiteWonStates = data_mat[:numWhiteWon]
        self.whiteLostStates = data_mat[numWhiteWon:]

        np.random.shuffle(self.whiteWonStates)
        np.random.shuffle(self.whiteLostStates)

        self.whiteWonStatesX = self.whiteWonStates[:self.sampleSize]
        self.whiteLostStatesX = self.whiteLostStates[:self.sampleSize]

        self.whiteWonStatesY = np.zeros((self.sampleSize,))
        self.whiteLostStatesY = np.ones((self.sampleSize,))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.sampleSize / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # We need to prepare batch for each index
        curr_batch_index = batch_size*index

        if self.sampleSize - curr_batch_index < 0:
            X1 = self.whiteWonStatesX[curr_batch_index:]
            X2 = self.whiteLostStatesX[curr_batch_index:]
            Y1 = self.whiteWonStatesY[curr_batch_index:]
            Y2 = self.whiteLostStatesY[curr_batch_index:]
        else:
            X1 = self.whiteWonStatesX[curr_batch_index:
                                      curr_batch_index+self.batch_size]
            X2 = self.whiteLostStatesX[curr_batch_index:
                                       curr_batch_index+self.batch_size]
            Y1 = self.whiteWonStatesY[curr_batch_index:
                                      curr_batch_index+self.batch_size]
            Y2 = self.whiteLostStatesY[curr_batch_index:
                                       curr_batch_index+self.batch_size]

        # 0 means (W, L), 1 means (L, W)
        allow_swap = np.random.randint(2)
        if allow_swap == 1:
            X = [X2, X1]
            Y = [Y2, Y1]
        else:
            X = [X1, X2]
            Y = [Y1, Y2]

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.whiteWonStates = data_mat[:numWhiteWon]
        self.whiteLostStates = data_mat[numWhiteWon:]
        np.random.shuffle(self.whiteWonStates)
        np.random.shuffle(self.whiteLostStates)
        self.whiteWonStatesX = self.whiteWonStates[:self.sampleSize]
        self.whiteLostStatesX = self.whiteLostStates[:self.sampleSize]


# GENERATOR - to generate data on the fly
dataGen = DataGenerator(batch_size, sampleSize)

# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

# combine the output of the two branches
combined = Concatenate()([dbn_model[0].output, dbn_model[1].output])

deep1 = Dense(deepChessLayers[0], activation="relu")(combined)
deep2 = Dense(deepChessLayers[1], activation="relu")(deep1)
deep3 = Dense(deepChessLayers[2], activation="relu")(deep2)
deep4 = Dense(deepChessLayers[3], activation="softmax")(deep3)

# our model will accept the inputs of the two branches and
# then output two values
deep_chess_model = Model(
    inputs=[dbn_model[0].input, dbn_model[1].input], outputs=deep4)
deep_chess_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

deep_chess_model.summary()

# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/

""" We are using fit_generator() since our data set is massive and 
    may not fit into RAM
    
    "As the name suggests, the .fit_generator  function assumes there is an 
    underlying function that is generating the data for it." 

    This is perfect as we are essentially generating new data set for every 
    epoch. See `Training DeepChess` section in the paper

    We need to create a Python generator functions and pass it into the 
    fit_generator function. 

    Generator functions allow you to declare a function that behaves like an iterator, i.e. it can be used in a for loop.
    https://wiki.python.org/moin/Generators

    And main tutorial:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Tutorial on Siamese Network and Keras:
    https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d
"""

deep_chess_model.fit_generator(
    generator=dataGen, use_multiprocessing=True, epochs=deep_chess_epochs)

# SAVE THE FINAL MODEL
timestr = time.strftime("%Y%m%d-%H%M%S")
model_filename = "deepchess-" + timestr + ".h5"
deep_chess_model.save(os.path.join("./models/", model_filename))