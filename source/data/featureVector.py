
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