import numpy as np
import tensorflow as tf

# ✅ 1. 저장된 모델 불러오기
sizex = 6 # 보드의 세로
sizey = 5 # 가로
model = tf.keras.models.load_model('./model/tictactoe_model_6x5.h5')  # 저장된 모델 파일을 불러옵니다.


new_board = np.array([
    [1, 2 ,2, 1, 0],
    [2, 1 ,0, 0, 2],
    [0, 0 ,0, 0, 2],
    [0, 0 ,0, 0, 0],
    [0, 0 ,0, 0, 0],
    [0, 0 ,0, 0, 0],
], dtype=np.float32)


new_board = np.expand_dims(new_board, axis=0) 
predictions = model.predict(new_board)
predictions_reshaped = predictions.reshape(sizex, sizey)

for x in range(sizex):
    for y in range(sizey):
        print(f"Cell (row : {x}, col: {y}) predictions: {predictions_reshaped[x, y]}")

empty_cells = []
for x in range(sizex):
    for y in range(sizey):
        if new_board[0, x, y] == 0:
            empty_cells.append((x, y, predictions_reshaped[x, y]))

if empty_cells:
    empty_cells.sort(key=lambda cell: cell[2], reverse=True)
    best_move = empty_cells[0]
    best_x, best_y, best_prob = best_move
    print(f"The best move is at position (row: {best_x}, col: {best_y}) with probability {best_prob}")
else:
    print("No empty cells available.")


# 이 아래로 보드를 그립니다.
print(predictions_reshaped)

