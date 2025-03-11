import numpy as np
import tensorflow as tf

# ✅ 1. 저장된 모델 불러오기
model = tf.keras.models.load_model('./model/tictactoe_model_3x3.h5')  # 저장된 모델 파일을 불러옵니다.

# 예측을 위한 새로운 보드 (3x3 보드 예시)
new_board_3x3 = np.array([
    [1, 0, 0],  # 1: X, 0: 빈칸, 2: O
    [0, 1, 0],
    [0, 0, -1]
], dtype=np.float32)

# 보드를 1D 배열로 flatten (모델이 1D 배열을 예상하는 경우)
flattened_board = new_board_3x3.flatten()  # (9,) 형태로 변환

# 보드를 3x3 크기의 보드에 대해 9개의 채널을 만들도록 수정합니다.
one_hot_board = np.zeros((3, 3, 9), dtype=np.float32)  # (3, 3, 9) 보드 생성

# 각 보드 칸을 채널별로 분리하여 값을 설정
for idx, value in enumerate(flattened_board):
    row = idx // 3
    col = idx % 3
    if value == 1:
        one_hot_board[row, col, 0] = 1  # X -> 첫 번째 채널에 1 할당
    elif value == 2:
        one_hot_board[row, col, 1] = 1  # O -> 두 번째 채널에 1 할당
    else:
        one_hot_board[row, col, 2] = 1  # 빈칸 -> 세 번째 채널에 1 할당

# 보드를 (1, 3, 3, 9) 형태로 reshape
one_hot_board = one_hot_board.reshape(1, 3, 3, 9)  # (1, 3, 3, 9) 형태로 변환

# 예측 수행
predictions = model.predict(one_hot_board)

# 예측 결과 출력
print("Predictions shape:", predictions.shape)  # (1, 81)로 예상되는 출력

# 예측된 확률 값 출력 (출력 크기: 81)
predicted_probabilities = predictions.reshape(9, 9)  # 출력 크기에 맞게 reshape

# 각 예측된 칸에 대해 9개의 확률을 출력
print("Predicted probabilities for each of the 9 possible moves:")
print(predicted_probabilities)

# 예측된 최적의 수 (확률이 가장 높은 값 찾기)
predicted_move = np.argmax(predicted_probabilities)
predicted_row = predicted_move // 9  # 9개의 칸에서 행 계산
predicted_col = predicted_move % 9   # 열 계산

print(f"Predicted best move: Row {predicted_row}, Column {predicted_col}")