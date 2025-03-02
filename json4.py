import json
import numpy as np
import glob
import os

# JSON 파일 경로 패턴
json_file_pattern = './data-*.json'

# JSON 파일 읽기
data = []
for json_file_path in sorted(glob.glob(json_file_pattern)):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data.extend(json.load(file))

if not data:
    print("🚨 JSON 파일을 찾을 수 없습니다! 경로를 확인하세요.")
    exit()

# 🟢 보드 상태를 3x3 배열로 변환 (CNN 대비)
def board_to_numeric(board_state):
    return np.array([[1 if cell == 'X' else (-1 if cell == 'O' else 0) for cell in row] for row in board_state])

# 🟢 승리한 플레이어가 둔 모든 수를 저장하는 함수
def find_all_winning_moves(history, result):
    if result == "Draw":
        return []  # 무승부인 경우 빈 리스트 반환
    return [move for move in history if move["player"] == result]  # 승리한 플레이어가 둔 모든 수 반환

# 🟢 데이터 변환 함수 (이긴 플레이어의 모든 수를 학습 데이터로 저장)
def generate_data(data):
    X = []  # 입력 데이터 (보드 상태)
    y = []  # 타겟 데이터 (이긴 플레이어가 둔 수)

    for item in data:
        result = item['result']
        winning_moves = find_all_winning_moves(item['history'], result)  # 모든 수 가져오기

        for move in winning_moves:
            board_state = board_to_numeric(move['boardState'])  # 보드 상태 변환
            X.append(board_state)
            y.append(move['row'] * 3 + move['col'])  # 0~8 위치 변환

    X = np.array(X).reshape(-1, 3, 3, 1)  # CNN 입력 형태 (num_samples, 3, 3, 1)
    y = np.array(y, dtype=np.int32)  # 정수 레이블로 변환

    print(f"✅ 변환된 데이터 개수: {len(X)}")
    return X, y

# 🟢 데이터 검증 함수
def check_data(X, y):
    for i in range(18):  # 샘플 2개 확인
        board = X[i].reshape(3,3)
        move = y[i]

        print(f"\n🔍 샘플 {i+1}:")
        print(board)
        print(f"👉 예측할 위치 (y): {move} ({move // 3}, {move % 3})")
        if board[move // 3, move % 3] != 0:
            print("⚠️ 오류! y 값이 빈칸이 아님!")

# 🟢 학습 데이터 생성
X, y = generate_data(data)

# 🟢 데이터 검증 (샘플 확인)
check_data(X, y)  # ✅ 추가: 데이터가 올바른지 확인

# 🟢 학습 데이터를 파일로 저장
if len(X) > 0:
    np.save('train_x_cnn.npy', X)
    np.save('train_y_cnn.npy', y)
    print(f"🎯 학습 데이터 저장 완료: train_x_cnn.npy ({X.shape}), train_y_cnn.npy ({y.shape})")

    print("\n🔍 데이터 샘플 확인:")
    print("입력 데이터 (X) 샘플:\n", X[0].reshape(3, 3))  # 3x3 형태로 출력
    print("출력 데이터 (y) 샘플:\n", y[0])
else:
    print("🚨 데이터가 비어 있습니다! JSON 파일을 확인하세요.")
