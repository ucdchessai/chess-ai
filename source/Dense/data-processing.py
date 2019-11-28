#Data preprocessing for Chess AI

import pandas as pd
import chess

DATA_PATH = '../data/games.csv'

def main():
    AllData = importData()
    CleanData = clean(AllData)
    boardData = convertToBoardStates(CleanData)
    boardData = OneHotEncode(boardData)
    print(boardData.to_csv())

def OneHotEncode(Data):
    y = Data["Winner"]
    Data.drop(columns="Winner", inplace=True)
    y = pd.get_dummies(y)
    Data = pd.concat([Data, y], axis=1)
    return Data


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

def clean(data):
    data = data[data["winner"] != 'draw']
    data = data[data["white_rating"] >= 1750]
    return data

def convertToBoardStates(Data):
    WinBin = {'black':0, 'white':1}
    columns = []
    for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        for j in ['1', '2', '3', '4', '5', '6', '7', '8']:
            columns.append(i + j)
    columns.append('Winner')
    num = 0
    boardData = pd.DataFrame(columns=columns)
    for j in Data['moves'].index:
        board = chess.Board()
        for i in Data['moves'][j].split(' '):
            board.push_san(i)
            winner = Data['winner'][j]
            boardData.loc[num] = convertToFeatureVector(board) + [WinBin[winner]]
            num += 1
    return boardData




def importData():
    data = pd.read_csv(DATA_PATH, sep=',')    #read the file
    data.drop(columns="id", inplace=True)
    data.drop(columns="rated", inplace=True)
    data.drop(columns="created_at", inplace=True)
    data.drop(columns="last_move_at", inplace=True)
    data.drop(columns="increment_code", inplace=True)
    data.drop(columns="white_id", inplace=True)
    data.drop(columns="opening_name", inplace=True)
    data.drop(columns="opening_ply", inplace=True)
    data.drop(columns="black_id", inplace=True)
    # data.drop(columns="last_move_at", inplace=True)
    # data.drop(columns="increment_code", inplace=True)
    # data.drop(columns="white_id", inplace=True)
    return data


main()
