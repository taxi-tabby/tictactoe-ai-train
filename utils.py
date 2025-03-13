import numpy as np
import tensorflow as tf

import math

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from keras.regularizers import l2


def longest_sequence(board):
    def check_sequence(arr):
        max_len = 0
        current_len = 0
        current_val = None
        for val in arr:
            if val == current_val and val != 0:
                current_len += 1
            else:
                current_val = val
                current_len = 1
            if current_len > max_len:
                max_len = current_len
        return max_len

    max_length = 0

    # Check rows
    for row in board:
        max_length = max(max_length, check_sequence(row))

    # Check columns
    for col in board.T:
        max_length = max(max_length, check_sequence(col))

    # Check diagonals (left to right)
    for offset in range(-board.shape[0] + 1, board.shape[1]):
        max_length = max(max_length, check_sequence(board.diagonal(offset)))

    # Check diagonals (right to left) - using np.fliplr to flip the board horizontally
    for offset in range(-board.shape[0] + 1, board.shape[1]):
        max_length = max(max_length, check_sequence(np.fliplr(board).diagonal(offset)))

    return max_length


def value_to_numeric(value):
    """
    보드의 값 ('X', 'O', 빈칸)을 숫자로 변환하는 함수
    'X' → 1, 'O' → 2, 빈칸 → 0
    :param value: 보드에 있는 값 ('X', 'O', 빈칸)
    :return: 변환된 숫자 값
    """
    if value == 'X':
        return 1
    elif value == 'O':
        return 2
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



def dynamicBatchSizeR(train_data, current_data_size, max_batch_size=256, min_batch_size=32):
    """
    현재 학습에 들어가는 데이터 크기와 전체 데이터 크기를 비교하여,
    배치 크기를 동적으로 계산합니다. 데이터가 많을수록 배치 크기가 작아지도록 설정.
    """
    total_samples = sum(train_data)  # 전체 데이터 샘플 수 (train_data 배열의 합)
    
    # 현재 학습에 사용되는 데이터 크기
    current_samples = current_data_size
    
    # 전체 데이터 크기와 현재 데이터 크기의 비율을 계산
    ratio = current_samples / total_samples
    
    # 배치 크기를 비례하는 방식으로 설정 (반비례로 설정)
    batch_size = int(max_batch_size * (1 / ratio))  # 비율에 반비례하게 설정
    
    # 배치 크기를 설정된 범위 내로 제한
    batch_size = max(min_batch_size, min(batch_size, max_batch_size))
    
    return batch_size



def dynamicBatchSize(train_data, current_data_size, max_batch_size=256, min_batch_size=36):
    """
    현재 학습에 들어가는 데이터 크기와 전체 데이터 크기를 비교하여,
    배치 크기를 동적으로 계산합니다. 데이터가 많을수록 배치 크기가 작아지도록 설정.
    """
    total_samples = len(train_data)  # 전체 데이터 샘플 수
    current_samples = current_data_size  # 현재 학습에 사용되는 데이터 크기

    # 현재 데이터의 크기가 전체 데이터 크기에 대해 차지하는 비율을 계산
    ratio = current_samples / total_samples

    # 배치 크기를 비율에 반비례하도록 설정 (샘플 수가 많을수록 배치 크기가 작아짐)
    batch_size = int((1 / ratio) * max_batch_size)

    # 배치 크기를 설정된 범위 내로 제한
    batch_size = max(min_batch_size, min(batch_size, max_batch_size))
    
    return batch_size




def create_model2(input_shape):
    """
    각 칸에 확률을 예측하는 CNN 모델 생성 함수.
    더 다양한 데이터와 모델 개선을 반영한 구조
    """
    model = Sequential([

        # 첫 번째 Convolutional 레이어 (필터 수: 32, 커널 크기: 3x3, 활성화 함수: ReLU)
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),  # MaxPooling으로 다운샘플링

        # 두 번째 Convolutional 레이어 (필터 수: 64, 커널 크기: 3x3, 활성화 함수: ReLU)
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),  # MaxPooling으로 다운샘플링

        # 세 번째 Convolutional 레이어 (필터 수: 128, 커널 크기: 3x3, 활성화 함수: ReLU)
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),  # MaxPooling으로 다운샘플링

        # 차원 축소 후 Flatten 레이어로 1D 벡터로 변환
        Flatten(),

        # Dense 레이어 (512개의 유닛, Dropout 적용)
        Dense(512, activation='relu'),
        Dropout(0.5),

        # Dense 레이어 (256개의 유닛, Dropout 적용)
        Dense(256, activation='relu'),
        Dropout(0.3),

        # Dense 레이어 (64개의 유닛, Dropout 적용)
        Dense(64, activation='relu'),
        Dropout(0.1),

        # 출력 레이어 (각 칸에 대한 확률을 예측, 소프트맥스 활성화 함수)
        Dense(input_shape[0] * input_shape[1], activation='softmax'),  # 3x3 크기 보드의 예측
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model




def create_model1(input_shape):
    model = Sequential([

        # 첫 번째 Conv 레이어
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),  # Pooling size와 padding 확인

        # 두 번째 Conv 레이어
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        # 세 번째 Conv 레이어
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),  # 성능 향상을 위해 두 번 쌓음
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        # Global Average Pooling을 사용하여 최종 특성 벡터를 얻음
        GlobalAveragePooling2D(),

        # Fully connected layers
        Dense(512, activation='relu',),  # L2 정규화 추가
        Dropout(0.5),
        Dense(256, activation='relu'),  # L2 정규화 추가
        Dropout(0.5),

        # 출력층
        Dense(input_shape[0] * input_shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
