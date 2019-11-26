import chess
import chess.engine
import chess.svg
import random
from IPython.display import display

import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow as tf 
import pandas as pd
import numpy as np

import os
import time
import math

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gamesToPlay = 25

# 0-Stockfish, 1-Crafty <--- CHANGE ENGINE HERE. DEFAULT: STOCKFISH
engine_num = 0

engines_names = ["Stockfish", "Crafty"]

nn = 0
nn_names = ["DeepChess", "SimpleDense"]
nn_fn_array = ["deepchess-69.h5", "simple_dense.h5"]
decision_string = []
svg_array=[]

# Higher the depth, the more moves it looks ahead.
# if depth=5, then it looks ahead 5 moves for every legal move

nn_depth = 4 # <--- DEPTH OF OUR ALPHABETA ALGORITHM
engine_depth = 4 # <--- DEPTH OF THE ENGINE

print("Loading Model")
model = tf.keras.models.load_model(nn_fn_array[nn])
print("Done Loading")

def chooseEngine(num):
    if num == 1:
        return chess.engine.SimpleEngine.popen_xboard("./crafty")
    return chess.engine.SimpleEngine.popen_uci("./stockfish_10_x64")

def countMoves(moves):
    length = 0
    for _ in moves:
        length += 1
    return length

def convertToFeatureVector(boardState):
    pieces = []
    encoding = {'P':1,'R':2,'N':3,'B':4,'Q':5,'K':6,'p':-1,'r':-2,'n':-3,'b':-4,'q':-5,'k':-6, '.':0}
    
    for i in range(64):
        piece = boardState.piece_at(i)
        if piece:
            piece = piece.symbol()
        else:
            piece = '.'
        pieces.append(encoding[piece])
        
    if(nn == 0):
        kingWhite = bool(boardState.castling_rights & chess.BB_H1)
        if(kingWhite):
            pieces.append(7)
        else:
            pieces.append(-7)

        queenWhite = bool(boardState.castling_rights & chess.BB_A1)
        if(queenWhite):
            pieces.append(8)
        else:
            pieces.append(-8)

        kingBlack = bool(boardState.castling_rights & chess.BB_H8)
        if(kingBlack):
            pieces.append(9)
        else:
            pieces.append(-9)

        queenBlack = bool(boardState.castling_rights & chess.BB_A8)
        if(queenBlack):
            pieces.append(10)
        else:
            pieces.append(-10)

        if(board.turn):
            pieces.append(11)
        else:
            pieces.append(-11)
    
    return pieces

def evaluateMove(move1, move2):    
    df1 = pd.DataFrame(data=[move1])
    df2 = pd.DataFrame(data=[move2])
    return model.predict([df1, df2])


def depthZeroSearch(board):
    ogBoard = board
    count = 0
    move1 = move2 = None
    board1 = board2 = None

    for move in ogBoard.legal_moves:
        if count == 0:
            move1 = move
        else:
            modBoard = ogBoard
            move2 = move

            modBoard.push(move1)
            move1Feature = convertToFeatureVector(modBoard)
            modBoard.pop()

            modBoard.push(move2)
            move2Feature = convertToFeatureVector(modBoard)
            modBoard.pop()
            strengthArr = evaluateMove(move1Feature, move2Feature)[0]
            if(strengthArr[0] < strengthArr[1]):
                move1 = move2
        count += 1
    return move1

# This is the modified AlphaBeta function as mentioned in the DeepChess.
# Not sure if I implemented it correctly, feel free to play around with it.
def comparisonAlphaBeta(board, depth, whitesTurn):
    if depth == 0 or board.is_game_over():
        return board
    
    board = board.copy()
    best_board = board

    cnt = 0
    mv1 = None
    prune = False
    for move in board.legal_moves:
        if cnt > 1:
            board.push(move)
            if whitesTurn:
                evaluated_pos = comparisonAlphaBeta(board, depth - 1, False)
            else:
                evaluated_pos = comparisonAlphaBeta(board, depth - 1, True)
            evaluated_move = convertToFeatureVector(evaluated_pos)
            board.pop()

            # Now we compare the moves with alpha position
            board.push(alpha)
            strengthAlpha = evaluateMove(convertToFeatureVector(board), evaluated_move)[0]
            board.pop()

            # Now we compare the moves with beta position
            board.push(beta)
            strengthBeta = evaluateMove(convertToFeatureVector(board), evaluated_move)[0]
            board.pop()
            
            beMove = True
            if whitesTurn:
                if(strengthAlpha[0] > strengthAlpha[1]):
                        best_board = evaluated_pos
                        alpha = move
                        beMove = False
                if(strengthBeta[0] < strengthBeta[1]):
                        break
                elif strengthBeta[0] >= strengthBeta[1] and beMove:
                    beta = move
            else:
                if(strengthAlpha[0] <= strengthAlpha[1]):
                        best_board = evaluated_pos
                        alpha = move
                        beMove = False
                if(strengthBeta[0] > strengthBeta[1]):
                        break
                elif strengthBeta[0] <= strengthBeta[1] and beMove:
                    beta = move
                    
        elif cnt == 1:
            board.push(mv1)
            b1 = board.copy()
            v1 = convertToFeatureVector(board)
            board.pop()

            board.push(move)
            b2 = board.copy()
            v2 = convertToFeatureVector(board)
            board.pop()

            strength = evaluateMove(v1, v2)[0]
            if strength[0] > strength[1]:
                alpha, beta = mv1, move
            else:
                alpha, beta = move, mv1
        else:
            mv1 = move
        cnt += 1

    return best_board

for _ in range(0, gamesToPlay):
    startTime = time.time()
    countTurns = 0
    board = chess.Board()
    engine = chooseEngine(engine_num)
    for i in range(0, 2):
        # If we make two moves random, we play as white
        moves = board.legal_moves
        moveCount = countMoves(moves)
        randomIndex = random.randrange(0,moveCount,1)

        count = 0
        for move in moves:
            if(count == randomIndex):
                board.push(move)
                break
            count += 1

    while not board.is_game_over():
        countTurns = countTurns + 1
#         if board.legal_moves.count() > 2: 
        board = comparisonAlphaBeta(board, nn_depth, True)
#         else:
#             board.push(depthZeroSearch(board))
                
        if(board.is_game_over()):
            break;
        
        result = engine.play(board, chess.engine.Limit(depth=engine_depth))
        board.push(result.move)
        countTurns = countTurns + 1
    
    endTime = time.time()
    svg_array.append(chess.svg.board(board=board))
    print_str = nn_names[nn] + ": " + board.result() + " :" + engines_names[engine_num] + ", Total Moves: " + str(countTurns) + ", Time: " + str(math.ceil(endTime-startTime)) + " seconds.\n"
    decision_string.append(print_str)
    print(print_str)
    engine.quit()

res_file = open("StockfishDepth4.txt","a+")
print('-------------------------')
res_file.write("-------------------------\n")
temp_str = "NN Depth: " + str(nn_depth) + ", " + engines_names[engine_num] +" Depth: " + str(engine_depth) + "\n"
res_file.write(temp_str)
res_file.write("-------------------------\n")
for i, st in enumerate(decision_string):
    display(svg_array[i])
    print(st)
    res_file.write(st)
    print('-------------------------------')
    
res_file.write("\n")
res_file.close()