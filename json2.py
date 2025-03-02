import json
import numpy as np
import glob

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

# 보드 상태를 3x3 배열로 변환
def board_to_numeric(board_state):
    numeric_board = []
    for row in board_state:
        numeric_board.extend([1 if cell == 'X' else (-1 if cell == 'O' else 0) for cell in row])
    return numeric_board

# 최적의 수만 추출하는 함수
def find_best_moves(history, result):
    if result == "Draw":
        return history  # 무승부일 경우 모든 수 포함
    return [move for move in history if move["player"] == result]

# 데이터 변환 함수
def generate_data(data):
    X = []  # 입력 데이터 (보드 상태)
    y = []  # 타겟 데이터 (이긴 수)

    for item in data:
        result = item['result']
        best_moves = find_best_moves(item['history'], result)

        for move in best_moves:
            board_state = move['boardState']
            numeric_board = board_to_numeric(board_state)  # 보드 상태를 숫자로 변환

            y.append(move['row'] * 3 + move['col'])  # 0~8 위치 변환
            X.append(numeric_board)  # 보드 상태 저장
                
    X = np.array(X)
    y = np.array(y)

    print(f"✅ 변환된 데이터 개수: {len(X)}")
    return X, y

# 학습 데이터 생성
X, y = generate_data(data)

# 학습 데이터를 파일로 저장
if len(X) > 0:
    np.save('train_x2.npy', X)
    np.save('train_y2.npy', y)
    print("🎯 학습 데이터 저장 완료: train_x2.npy, train_y2.npy")
else:
    print("🚨 데이터가 비어 있습니다! JSON 파일을 확인하세요.")
