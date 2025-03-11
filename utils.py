import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping

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
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(64, (3, 3), activation='relu'),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(input_shape[0] * input_shape[1], activation='softmax')  # 출력 크기
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
