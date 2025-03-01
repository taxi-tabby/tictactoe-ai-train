# Tic-Tac-Toe Model

## Overview
This repository contains a project for training a Tic-Tac-Toe AI model using machine learning. The process involves data preprocessing, training the model, and using it for predictions. Below are the steps to run the project.

## Setup Instructions

### 1. Unzip the Data
After unzipping `data.zip`, you will get the following JSON files:

- `data1.json`
- `data2.json`
- `data3.json`
- `data4.json`

### 2. Data Normalization
In the same directory, execute `json.py` to normalize the data.  
This will generate the following files in the directory:

- `train_x.npy`
- `train_y.npy`

### 3. Model Training
If `train_x.npy` and `train_y.npy` are present, you can run `train.py` to train the model.  
After training, a model file called `tictactoe_model.h5` will be created.

## Prediction

Once the model (`tictactoe_model.h5`) is trained, you can use it for predictions.

### Input Format
The model expects an input array in the following format (a 1D array of length 9 representing the current state of the board):
- `[1, -1, 0, 0, 1, -1, 0, 0, 1]`

```python
- `X = 1`
- `O = -1`
- `0` represents an empty space.
```
### Output
The model will return the best move (a single number between 0 and 8), which represents the best position to place the next marker (either X or O).

Example:

- `best_move = model.predict([1, -1, 0, 0, 1, -1, 0, 0, 1])`

This will return the index of the best move on the board.

## Requirements

- Python 3.x
- TensorFlow
- NumPy

You can install the required libraries using the following command:

