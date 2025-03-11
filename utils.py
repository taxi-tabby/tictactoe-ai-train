import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

def value_to_numeric(value):
    """
    보드의 값 ('X', 'O', 빈칸)을 숫자로 변환하는 함수
    'X' → 1, 'O' → -1, 빈칸 → 0
    :param value: 보드에 있는 값 ('X', 'O', 빈칸)
    :return: 변환된 숫자 값
    """
    if value == 'X':
        return 1
    elif value == 'O':
        return -1
    else:
        return 0  # 빈칸은 0


def board_to_numeric(board_state, target_rows, target_cols):
    """
    보드 상태를 숫자로 변환하여 target 크기로 맞춤
    :param board_state: 보드 상태 (리스트 형태)
    :param target_rows: 보드의 목표 행 수
    :param target_cols: 보드의 목표 열 수
    :return: 변환된 숫자 배열
    """
    rows = len(board_state)
    cols = len(board_state[0]) if rows > 0 else 0

    numeric_board = np.zeros((target_rows, target_cols), dtype=float)

    for i in range(rows):
        for j in range(cols):
            numeric_board[i][j] = value_to_numeric(board_state[i][j])

    return numeric_board

def load_data(board_size):
    X = np.load(f'./npy/train_x_{board_size[0]}x{board_size[1]}.npy')
    y = np.load(f'./npy/train_y_{board_size[0]}x{board_size[1]}.npy')
    return X, y

def create_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(input_shape[0] * input_shape[1], activation='softmax')  # 출력 크기
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
