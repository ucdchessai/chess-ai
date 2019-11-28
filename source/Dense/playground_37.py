import anti_alexnet
from   chess import Board
import math
from   numpy import ones
from   pandas import read_csv
from   PlayerAi import PlayerAi
from   random import choice
from   random import seed
from   random import shuffle

def main():
    model_file_name = 'simple_dense.h5'
    train_simple_dense(model_file_name, iterations=0)
    model = anti_alexnet.load_model(model_file_name)

    # Look at what the model's first move is.
    board = Board()
    ai = PlayerAi(model)
    print(board)
    while (not(board.is_game_over())):
        move = ai.get_move(board, decisiveness=16, verbose=True)
        board.push(move)
        print(board)
        moves = [move for move in board.legal_moves]
        board.push(choice(moves))
        print(board)


def get_data_set(training_samples, testing_samples=0):
    """
    Returns a list containing the training set inputs, training set outputs,
    test set inputs, and test set outputs.

    ``training_samples`` - The number of training samples to retrieve.
    ``testing_samples`` - The number of testing samples to retrieve.
    """

    n = training_samples + testing_samples

    data = loadData()

    data = data.sample(n=n).values

    indices = [x for x in range(n)]
    shuffle(indices)
    indices_training = indices[:training_samples]
    indices_testing = indices[training_samples:]

    training_set = data[indices_training]
    testing_set = data[indices_testing]

    return [training_set[:,:64:], training_set[:,64:], testing_set[:, :64:],
            testing_set[:, 64:]]


def get_simple_dense():
    """
    Returns a simple densely-connected ANN with 8 hidden layers and 32 nodes per
    hidden layer.
    """
    return anti_alexnet.get_dense(hidden_layer_count=8, layer_node_count=32)


def k_fold(n, data_count = 340000, batch_size=32, epochs=10):
    all_X, all_Y, _, _ = get_data_set(data_count)

    scramble = list(range(data_count))
    shuffle(scramble)

    splits = [int(float(data_count) * i/n) for i in range(n + 1)]

    train_Xs = []
    train_Ys = []
    test_Xs = []
    test_Ys = []
    for i in range(len(splits) - 1):
        train_indices = scramble[:splits[i]] + scramble[splits[i+1]:]
        test_indices = scramble[splits[i]:splits[i+1]]

        train_Xs.append(all_X[train_indices,:])
        train_Ys.append(all_Y[train_indices,:])
        test_Xs.append(all_X[test_indices,:])
        test_Ys.append(all_Y[test_indices,:])

    gen_acc = [0] * (epochs + 1)
    for i in range(n):
        print('*** Fold {}/{} ***'.format(i + 1, n))
        model = get_simple_dense()

        error = model.evaluate(test_Xs[i], test_Ys[i], verbose=0)
        gen_acc[0] += math.e**(-error)
        for epoch in range(epochs):
            print('\tEpoch {}/{}'.format(epoch + 1, epochs))
            model.fit(train_Xs[i], train_Ys[i], batch_size=batch_size, epochs=1, verbose=0)
            error = model.evaluate(test_Xs[i], test_Ys[i], verbose=0)
            gen_acc[epoch + 1] += math.e**(-error)

    gen_acc = [x / n for x in gen_acc] # Divide to get average.

    return gen_acc


def loadData():
    data = read_csv('ChessData.csv', sep=',')
    data.drop(columns="Unnamed: 0", inplace=True)
    return data


def param_sweep():
    """
    Performs a parameter sweep on the number of hidden layers and number of
    nodes per hidden layer. A lxmxn matrix is displayed for the training and
    test errors, where l is the number of epochs plus one, m is the number of
    hidden layer parameters, and n is the number of layer node parameters. The
    sweep goes through 5 epochs, the hidden layer counts in [2, 4, 6, 8], and
    the layer node counts in [4, 8, 16, 32, 64].
    """

    batch_size = 100
    epochs = 5

    # Initialize some variables.
    hidden_layer_counts = list(range(2, 9, 2))
    layer_node_counts = [2**x for x in range(2, 7)]
    errors = ones((epochs + 1, len(hidden_layer_counts), len(layer_node_counts)))
    losses = ones((epochs + 1, len(hidden_layer_counts), len(layer_node_counts)))

    [X, Y, X_test, Y_test] = get_data_set(100000, 10000)

    # Perform grid search on number of hidden layers and number of nodes per
    # hidden layer.
    for hidden_layer_count in hidden_layer_counts:
        for layer_node_count in layer_node_counts:
            model = anti_alexnet.get_dense(hidden_layer_count, layer_node_count,
                    activation_hidden='leakyrelu')
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
                #model.save('model_dense_' + str(hidden_layer_count) + '_' +
                #        str(layer_node_count) + '.h5')

    print('Testing errors:\n{0}'.format(errors))
    print('Training losses:\n{0}'.format(losses))
    return [errors, losses]


def train_simple_dense(file_name, iterations=None, training_sample_count=300000,
        test_sample_count=40000, batch_size=100, epoch_steps=1, save_best=False,
        by_performance=False, verbose=1):
    """
    Trains a simple densely-connected neural network and saves the model.

    ``file_name`` - If this file exists, the function will try to load it as a
    model and train it. If it fails or if the file doesn't exist, a new simple
    dense model is created. The model will be saved with this file name after
    every iteration.
    ``iterations`` - The number of iterations to train for. An iteration is
    defined as completion of ``epoch_steps`` number of epochs. After each
    iteration, the test error is displayed. If ``iterations`` is ``None``, the
    number of iterations is infinite.
    ``training_sample_count`` - The number of board state samples to use for
    training. The sum of this and ``test_sample_count`` must not be greater than
    the number of samples that can be obtained with ``get_data_set()``.
    ``test_sample_count`` - The number of board state samples to use for
    testing.
    ``batch_size`` - The batch size for stochastic gradient descent.
    ``epoch_steps`` - The number of epochs per iteration.
    ``save_best`` - Only re-save if the test loss is better.
    ``by_performance`` - Name the file based on performance.
    """

    # Load or create simple dense ANN.
    try:
        model = anti_alexnet.load_model(file_name)
        print('Loaded model from {0}.'.format(file_name))
    except:
        print('Creating new model.')
        model = get_simple_dense()
        model.save(file_name)

    # Get training and testing sets.
    [X, Y, X_test, Y_test] = get_data_set(training_sample_count,
            test_sample_count)

    # Train.
    if (iterations is None):
        inc = 0
        iterations = 1
    else:
        inc = 1
    i = 0
    best_loss = model.evaluate(X_test, Y_test, verbose=0)
    while (i < iterations):
        model.fit(X, Y, batch_size=batch_size, epochs=epoch_steps,
                verbose=verbose)
        error = model.evaluate(X_test, Y_test, verbose=0)
        if ((verbose > 0) & (test_sample_count > 0)):
            print('Testing error: {0}\n'.format(error))
        if (not(math.isnan(error))):
            unique_name = file_name
            if (by_performance):
                name_split = file_name.split(sep='.')
                performance = math.e**(-error)
                name_split[0] += '_r' + str(int(round(100. * performance)))
                unique_name = '.'.join(name_split)
            if (not(save_best) or (error < best_loss)):
                best_loss = min(best_loss, error)
                model.save(unique_name)
        i += inc


if __name__ == '__main__':
    main()

