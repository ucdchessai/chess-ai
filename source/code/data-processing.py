#Data preprocessing for Chess AI

import pandas as pd
import chess

DATA_PATH = '../data/games.csv'

def main():
    importData()

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

    print(data)
    return data


main()
