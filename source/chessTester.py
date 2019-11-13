import chess
import random

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
    return pieces

def evaluateMove(move):
	return 1



if (random.random() < .5):
	#stockfish makes move first
	dummyVar = 2
currentPlayer = 0
#while(notCheckmate and notStalemate):
board = chess.Board()
moves = board.legal_moves
moveStrength = 0
moveToMake = 0
for move in moves:
	board.push(move)
	moveFeature = convertToFeatureVector(board)

	strength = evaluateMove(move)
	if strength > moveStrength:
		moveToMake = move
	newMove = board.pop()
board.push(move)
	#let the other team make their move

