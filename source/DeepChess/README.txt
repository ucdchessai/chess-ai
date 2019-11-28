HOW TO TRAIN:
-------------

0) Make sure you have `data` and `models` in your directory.
1) Make sure you hava `white(somthing).npy` and `black(something).npy` in the
    `./data/` directory.
    a) if you are training the model with 773 inputs from data extracted from
        OUR DATA SET, then use the files `blackBit.npy` and `whiteBit.npy`
    b) if you are training the model with 773 inputs from data extracted from
        data set mentioned in the paper, then use the files 
        `black_dp.npy` and `black_dp.npy`
    Datasets can be found in Google Drive:
    https://drive.google.com/drive/u/1/folders/1mmu_TuPDrpijReAMeUr7dY4bgz4p-n8V
    
1.5) Make sure to update what dataset you want to use in `dbn.py` and `deepchess.py`
2) Run `python3 dbn.py`. Let it finish. This automatically stores a file called
    `dbn-model.h5` in `models` directory.
3) Run `python3 deepchess.py`. This will load the `dbn-model.h5` from the 
    `models` directory and begins training the DeepChess layers. The final 
    trained model will be stored in the `models` directory with name:
    `deepchess-(timestamp).h5`.
