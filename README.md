# Neural Network-Based Chess Agents

## Description

This repository contains the design and implementation of two chess players using feed-forward neural networks (FFNNs) and a combination of deep belief networks and Siamese networks based on the DeepChess architecture. These networks were trained on publicly available game data from Lichess.org, and are simple enough to be trained on a personal computer. Our fully trained models performed competently enough to reliably beat a casual human player, and our chess player modelled on the DeepChess architecture comes close to being on par with open-source chess engines StockFish and Crafty.

## Dependencies

* Operating Systems: Verified on Ubuntu 18.04
* Programming Languages: Python (version >= 3.7)
* Libraries: Keras, TensorFlow, pandas, numpy, python-chess, Jupyter Notebook

## How to Use

`/source` contains the source code for our chess players. 

* `/data` contains the Lichess dataset in `games.csv` as well as the script files for data preparation.
* `/python-chess-0.28.3` contains the source code for open-source chess interface `python-chess`.
* `/Dense` contains the model for our feed-forward neural network.
* `/DeepChess` contains the model for our deep belief network / Siamese network based on the DeepChess architecture.
* `Tester` contains the testing code pitting our models against chess engines StockFish and Crafty, as well as an interface for human players to play against our models.

### Data Preparation: 

Open up `DataProcessing.ipynb` in `/source/data` on Jupyter Notebook and run the code.

### Training:

To train our feed-forward neural network, run `python3 train.py` in `/source/Dense`.

To train our DeepChess-based neural network, first run `python3 dbn.py` and then `python3 deepchess.py` in `/source/DeepChess`.

### Testing:

To test our networks against other engines (executables provided in `/source/Tester/engines`), open up `PlayChess.ipynb` in Jupyter Notebook and change the parameters for the input model and the opponent engine. Then run the code.

To play against our models using the `python-chess` interface, run `python3 PlayAgainst.py` in `/source/Tester`.

## Citation

Our report can be found in the directory `/report`.

## License

See the LICENSE file for license rights and limitations (MIT).

## Acknowledgement

This work was part of a course project for ECS 171 Machine Learning at the University of California, Davis, taught by Assoc. Prof. Ilias Tagkopoulos. It was largely influenced and adapted from the DeepChess paper (David, Netanyahu, Wolf) (https://arxiv.org/abs/1711.09667).

Contributors: Alexander Ricalde, Ezekiel Morton, Joel Lee, Lawrence Lee, Muhammed Halbutogari, Nick Van, Nikhila Thota, Pavit Bath, Vivienne Chiang, Yoosuf Shafi