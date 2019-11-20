import chess
import statistics

class PlayerAi:
    """
    Represents a chess-playing AI.
    """

    def __init__(self, model):
        """
        ``model`` - A keras model which predicts the fitness of a board state
        given a feature vector representing the board.
        """

        self.__model = model


    def get_move(self, board, verbose=False):
        """
        Given a board state, returns the best next move according to the model.
        """

        fitnesses = []
        max_fitness = None;
        best_move = None;

        for move in board.legal_moves:
            # Find next board state.
            end_board = board.copy()
            end_board.push(move)
            pieces = self.__board_to_sample(end_board)

            # Predict fitness.
            fitness = self.__model.predict([pieces])[0][0]
            fitnesses.append(fitness)

            # Update best move if needed.
            if ((max_fitness is None) or (max_fitness < fitness)):
                max_fitness = fitness
                best_move = move

        if (verbose):
            print('Number of moves: {0}'.format(len(fitnesses)))
            print('Mean fitness: {0}'.format(statistics.mean(fitnesses)))
            # For some reason, statistics.stdev() fails.
            #print('Standard deviation: {0}'.format(statistics.stdev(fitnesses)))
            print('Minimum fitness: {0}'.format(min(fitnesses)))
            print('Maximum fitness: {0}'.format(max(fitnesses)))
            print('Fitness = {0}'.format(max_fitness))

        return best_move


    def __board_to_sample(self, board):
        """
        Converts a chess Board into a feature vector recognized by the model.
        """

        pieces = [0] * 64
        encoding = {'P':1,'R':2,'N':3,'B':4,'Q':5,'K':6,'p':-1,'r':-2,'n':-3,'b':-4,'q':-5,'k':-6}

        for i, piece in board.piece_map().items():
            pieces[i] = encoding[piece.symbol()]

        return pieces

