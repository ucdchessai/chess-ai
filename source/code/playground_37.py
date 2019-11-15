import anti_alexnet
import chess
import pandas as pd
from   PlayerAi import PlayerAi

def main():
    model_file_name = 'anti_alexnet.h5'
    try:
        model = anti_alexnet.load_model(model_file_name)
    except:
        model = anti_alexnet.get_new()
        [X, Y] = get_training_set(10000)
        model.fit(X, Y, batch_size=8, epochs=80, verbose=1, workers=3,
                use_multiprocessing=True)
        model.save(model_file_name)

    board = chess.Board()
    ai = PlayerAi(model)
    move = ai.get_move(board, verbose=False)
    board.push(move)
    print(board)


def get_training_set(n):
    data = loadData()
    y = data["Winner"]
    data.drop(columns="Winner", inplace=True)
    y = pd.get_dummies(y)
    data = pd.concat([data, y], axis=1)

    train = data.sample(n=n)
    train = train.values

    return [train[:,:64:], train[:,64:]]


def loadData():
    data = pd.read_csv('NewChessData.csv', sep=',')
    data.drop(columns="Unnamed: 0", inplace=True)
    return data


if __name__ == '__main__':
    main()

