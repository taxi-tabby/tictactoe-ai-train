# Tic-Tac-Toe Model

## Overview
This repository contains a project for training a Tic-Tac-Toe AI model using machine learning. The process involves data preprocessing, training the model, and using it for predictions. Below are the steps to run the project.

The goal of this model is to develop a predictive system that can handle a range of board sizes, from 3x3 to 9x9, with winning conditions varying from 3 to 9.

I look forward to seeing a simple game like Tic Tac Toe evolve into something more challenging and fun. I've seen some games increase difficulty by enlarging the board or changing the winning conditions. However, they often end up following predictable patterns that make the gameplay mechanical. This model aims to create more progressive difficulty and keep the game engaging.

I hope that the dataset is always shared with each other through git
I used tensorflow to make web implantation easier


## Notices and Challenges
(2025-03-13) 

Currently, I am facing difficulties with attempting model ensembling due to a lack of technical expertise. Additionally, the time required for data preprocessing and training is significantly larger than anticipated.




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

### 2.1 Data Verification
You can verify the normalized data using `npy_check7.py`. This script generates a report in the form of an image, which will be saved in the `/npy_report` folder.


### 3. Model Training
If `train_x___.npy` and `train_y___.npy` are present, you can run `train7.py` to train the model.  
After training, a model file called `./model/~~~.h5` will be created.

## Prediction

Once the model (`./model/~~~.h5`) is trained, you can use it for predictions.

### Input Format
The model expects an input array in the following format (a 1D array of length 9 representing the current state of the board):
- `arr (3, 3)`

```python
- `X = 1`
- `O = 2`
- `0` represents an empty space.
```


### Output
Each model returns the most advantageous move for each board state, according to its size and format.

Example:

- `board_rates = model.predict([[1, -1, 0, ...], [1, -1, 0, ...], [1, -1, 0, ...] ...])`

This will return the index of the best move on the board.

## Requirements

- Python 3.x
- TensorFlow
- NumPy

You must install the required libraries ezpz


## Expectations and objectives and future expectations of how the model works
The model learns all forms of board configuration to predict the probability of the most likely movement on the board at the time to move to victory.

For now, I'm making predictions about the board composition at a moment
In the future, I hope the performance will improve enough to learn all the game flows and make predictions about the future.

------
[GitHub Game Repository](https://github.com/taxi-tabby/tictactoe-ai-game)




------------------
Latest data info (2025-03-13)
![Training Data Normalization Report](./npy_report/npy_report_20250313_005721.png)


