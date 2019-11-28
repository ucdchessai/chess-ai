import chess.pgn
import chess
import random
import numpy as np
import pandas as pd

# Open data
dataPath = "data/games.csv"

numWhiteMoves = 1000000
numBlackMoves = 1000000


def getValidMoves(game):
    validMoves = []

    # Iterate over moves in game
    for i, move in enumerate(game.mainline_moves()):
        # Filter out first five moves and captures
        # These are filtered according to the methodology presented in the deepchess paper
        if(not game.board().is_capture(move) and (i >= 5)):
            # Append the move index to the validMoves list
            validMoves.append(i)

    return validMoves


def convertToFeatureVector(boardState):
    pieces = []
    encoding = {'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
                'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6, '.': 0}

    for i in range(64):
        piece = boardState.piece_at(i)
        if piece:
            piece = piece.symbol()
        else:
            piece = '.'
        pieces.append(encoding[piece])

    board = boardState

    kingWhite = bool(board.castling_rights & chess.BB_H1)
    if(kingWhite):
        pieces.append(7)
    else:
        pieces.append(-7)

    queenWhite = bool(board.castling_rights & chess.BB_A1)
    if(queenWhite):
        pieces.append(8)
    else:
        pieces.append(-8)

    kingBlack = bool(board.castling_rights & chess.BB_H8)
    if(kingBlack):
        pieces.append(9)
    else:
        pieces.append(-9)

    queenBlack = bool(board.castling_rights & chess.BB_A8)
    if(queenBlack):
        pieces.append(10)
    else:
        pieces.append(-10)

    if(board.turn):
        pieces.append(11)
    else:
        pieces.append(-11)

    return pieces


def addMoves(game, moveArray, moveIndex):
    # Retrieve all vandidates for valid moves from the game
    # Candidates are moves that are not the first 5 and are not captures
    validMoves = getValidMoves(game)

    # List to store 10 randomly selected moves
    selectedMoves = []
    for i in range(10):
        if(not validMoves):
            break

        # Select move randomly, remove from valid moves
        move = random.choice(validMoves)
        validMoves.remove(move)
        selectedMoves.append(move)

    #print(selectedMoves)

    # Instantiate a new chess board
    board = chess.Board()
    moveCount = 0
    for i, move in enumerate(game.mainline_moves()):
        # Push new move to board
        board.push(move)

        # Break if maximum number of moves already reached
        if(moveIndex >= moveArray.shape[0]):
            break

        # Check if the current move is one of the selected moves
        if(i in selectedMoves):
            moveArray[moveIndex] = convertToFeatureVector(board)
            moveIndex += 1

    return moveIndex

# iterateOverData
# Iterates over the provided pgn file and extracts 10 random moves.
# The data is stored in numpy arrays
# Continues iterating until end of file or until the desired number of boards for each color win has been reached


def iterateOverData():

    # Initialize numpy arrays to store white and black moves
    whiteMoves = np.zeros((numWhiteMoves, 69))
    blackMoves = np.zeros((numBlackMoves, 69))

    # White and black move counts store how many white and black moves have been stored
    whiteMoveIndex = 0
    blackMoveIndex = 0
    count = 0

    # Openfile containing chess game data
    games = pd.read_csv(dataPath)
    whiteGames = games.query("winner == 'white'")
    #blackGames = games.filter(like='black', columns=["winner"])
    #print(whiteGames)
    for index, game in whiteGames.iterrows():
        #print(row['c1'], row['c2'])
        #game = chess.pgn.read_game(game["moves"])
        #print(game["moves"])
        board = chess.Board()
        for move in game["moves"].split():
            board.push_san(move)
        chessGame = chess.pgn.Game().from_board(board)

        if(whiteMoveIndex < numWhiteMoves):
            whiteMoveIndex = addMoves(chessGame, whiteMoves, whiteMoveIndex)
        else:
            break

    blackGames = games.query("winner == 'black'")
    for index, game in blackGames.iterrows():
        board = chess.Board()
        for move in game["moves"].split():
            board.push_san(move)
        chessGame = chess.pgn.Game().from_board(board)

        if(blackMoveIndex < numBlackMoves):
            blackMoveIndex = addMoves(chessGame, blackMoves, blackMoveIndex)
        else:
            break
    return whiteMoves, blackMoves


print("Starting")
white, black = iterateOverData()

np.save("../data/whiteWin.npy", white)
np.save("../data/blackWin.npy", black)

print("Done")
