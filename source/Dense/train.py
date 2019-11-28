#training the Neural Network

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import metrics


def main():
    Data = loadData()

    train = Data.sample(n=200000)
    test = Data.drop(train.index)
    train = train.values
    test = test.values

    ytrain = train[:,64:]
    xtrain = train[:,:64:]
    ytest = test[:,64]
    xtest = test[:,:64]

    model = getModel()


    model.fit(xtrain, ytrain, batch_size=128, epochs=10, verbose=1)


def loadData():
    data = pd.read_csv('ChessData.csv', sep=',')
    data.drop(columns="Unnamed: 0", inplace=True)
    return data


def getModel():
    """Get model function for special functions like relu and softmax"""
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequential()
    #first hidden
    model.add(Dense(64, input_dim=64, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #second hidden
    model.add(Dense(64, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #third hidden
    model.add(Dense(64, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #fourth hidden
    model.add(Dense(64, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #fith hidden
    model.add(Dense(32, input_dim=64, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #sxth hidden
    model.add(Dense(32, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #seventh hidden
    model.add(Dense(32, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #eighth hidden
    model.add(Dense(32, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #ninth hidden
    model.add(Dense(16, input_dim=64, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #tenth hidden
    model.add(Dense(16, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #eleventh hidden
    model.add(Dense(16, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #twelth hidden
    model.add(Dense(16, activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    #output
    model.add(Dense(2, activation='sigmoid',
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
    model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=[keras.metrics.CategoricalAccuracy()])

    return model


main()
