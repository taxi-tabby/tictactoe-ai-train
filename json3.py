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

# 🟢 최적의 수만 추출하는 함수
def find_best_moves(history, result):
    return history if result == "Draw" else [move for move in history if move["player"] == result]

# 🟢 데이터 변환 함수
def generate_data(data):
    X = []  # 입력 데이터 (보드 상태)
    y = []  # 타겟 데이터 (이긴 수)

    for item in data:
        result = item['result']
        best_moves = find_best_moves(item['history'], result)

        for move in best_moves:
            board_state = board_to_numeric(move['boardState'])  # 보드 상태 변환
            X.append(board_state)
            y.append(move['row'] * 3 + move['col'])  # 0~8 위치 변환

    X = np.array(X).reshape(-1, 3, 3)  # CNN 대비하여 3x3 형태 유지
    y = np.array(y)

    print(f"✅ 변환된 데이터 개수: {len(X)}")
    return X, y

# 🟢 학습 데이터 생성
X, y = generate_data(data)

# 🟢 원-핫 인코딩 옵션 (다중 분류 모델 대비)
one_hot = True
if one_hot:
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y, num_classes=9)  # 0~8을 원-핫 벡터로 변환

# 🟢 학습 데이터를 파일로 저장
if len(X) > 0:
    np.save('train_x3.npy', X)
    np.save('train_y3.npy', y)
    print(f"🎯 학습 데이터 저장 완료: train_x3.npy ({X.shape}), train_y3.npy ({y.shape})")

    # 🟢 데이터 검증 (샘플 출력)
    print("\n🔍 데이터 샘플 확인:")
    print("입력 데이터 (X) 샘플:\n", X[0])
    print("출력 데이터 (y) 샘플:\n", y[0])
else:
    print("🚨 데이터가 비어 있습니다! JSON 파일을 확인하세요.")
