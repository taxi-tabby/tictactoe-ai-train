# Tic-Tac-Toe Model

## Overview
This repository contains a project for training a Tic-Tac-Toe AI model using machine learning. The process involves data preprocessing, training the model, and using it for predictions. Below are the steps to run the project.

## Setup Instructions

### 1. Get Training Data
You can download the training data from the following link:
[TicTacToe Training Data generator](https://smallbrain-labo.work/game/train/tictactoe)

(The training data is provided in the data folder)

### 2. Data Normalization
In the same directory, execute `json7.py` to normalize the data.  
This will generate the following files in the directory:

- `./npy/train_x_{col}x{row}_{winning condition length}.npy`
- `./npy/train_y_{col}x{row}_{winning condition length}.npy`

### 3. Model Training
If `train_x___.npy` and `train_y___.npy` are present, you can run `train7.py` to train the model.  
After training, a model file called `./model/~~~.h5` will be created.

## Prediction

Once the model (`./model/~~~.h5`) is trained, you can use it for predictions.

### Input Format
The model expects an input array in the following format (a 1D array of length 9 representing the current state of the board):
- `[[1, 2, 0, 0, 1, 2, 0, 0, 1]]`

```python
- `X = 1`
- `O = 2`
- `0` represents an empty space.
```


### Output
Each model returns the most advantageous move for each board state, according to its size and format.

Example:

- `best_move = model.predict([[1, -1, 0, ...], [1, -1, 0, ...], [1, -1, 0, ...] ...])`

This will return the index of the best move on the board.

## Requirements

- Python 3.x
- TensorFlow
- NumPy

You can install the required libraries using the following command




------
[GitHub Game Repository](https://github.com/taxi-tabby/tictactoe-ai-game)






