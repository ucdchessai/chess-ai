import anti_alexnet
import chess
import numpy as np
import pandas as pd
from   PlayerAi import PlayerAi
import random

def main():
    model_file_name = 'anti_alexnet.h5'
    try:
        model = anti_alexnet.load_model(model_file_name)
    except:
        model = anti_alexnet.get_new()
        [X, Y] = get_data_set(10000)
        model.fit(X, Y, batch_size=8, epochs=80, verbose=1, workers=3,
                use_multiprocessing=True)
        model.save(model_file_name)

    board = chess.Board()
    ai = PlayerAi(model)
    move = ai.get_move(board, verbose=False)
    board.push(move)
    print(board)


def get_data_set(training_samples, testing_samples=0):
    n = training_samples + testing_samples

    data = loadData()

    data = data.sample(n=n).values

    indices = [x for x in range(n)]
    random.shuffle(indices)
    indices_training = indices[:training_samples]
    indices_testing = indices[training_samples:]

    training_set = data[indices_training]
    testing_set = data[indices_testing]

    return [training_set[:,:64:], training_set[:,64:], testing_set[:, :64:],
            testing_set[:, 64:]]


def loadData():
    data = pd.read_csv('ChessData.csv', sep=',')
    data.drop(columns="Unnamed: 0", inplace=True)
    return data


def param_sweep():
    import math

    batch_size = 100
    epochs = 5

    # Initialize some variables.
    hidden_layer_counts = list(range(2, 9, 2))
    layer_node_counts = [2**x for x in range(2, 7)]
    errors = np.ones((epochs + 1, len(hidden_layer_counts), len(layer_node_counts)))
    losses = np.ones((epochs + 1, len(hidden_layer_counts), len(layer_node_counts)))

    [X, Y, X_test, Y_test] = get_data_set(100000, 10000)

    # Perform grid search on number of hidden layers and number of nodes per
    # hidden layer.
    for hidden_layer_count in hidden_layer_counts:
        for layer_node_count in layer_node_counts:
            model = anti_alexnet.get_dense(hidden_layer_count, layer_node_count,
                    activation_hidden='sigmoid')
            for epoch in range(epochs + 1): # Allows inspection of progress.
                if (epoch != 0): # Initial state.
                    model.fit(X, Y, batch_size=batch_size, epochs=epochs,
                            initial_epoch=(epoch - 1), verbose=0)

                # Testing error.
                error = model.evaluate(X_test, Y_test, verbose=0)
                errors[epoch][hidden_layer_counts.index(hidden_layer_count)]\
                        [layer_node_counts.index(layer_node_count)] = error

                # Traiing error.
                loss = model.evaluate(X, Y, verbose=0)
                losses[epoch][hidden_layer_counts.index(hidden_layer_count)]\
                        [layer_node_counts.index(layer_node_count)] = loss

                if (math.isnan(error)):
                    print('Warning: nan error for model with {0} hidden' +
                            'layers and {1} nodes per hidden layer.'
                            .format(hidden_layer_count, layer_node_count))
                    break
                model.save('model_dense_' + str(hidden_layer_count) + '_' +
                        str(layer_node_count) + '.h5')

    print('Testing errors:\n{0}'.format(errors))
    print('Training losses:\n{0}'.format(losses))
    return [errors, losses]


if __name__ == '__main__':
    main()

