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

# 보드 상태를 3x3 배열로 변환
def board_to_numeric(board_state):
    numeric_board = []
    for row in board_state:
        numeric_board.extend([1 if cell == 'X' else (-1 if cell == 'O' else 0) for cell in row])
    return numeric_board

# 데이터 변환 함수
def generate_data(data):
    X = []  # 입력 데이터 (보드 상태)
    y = []  # 타겟 데이터 (이긴 수)

    for item in data:
        result = item['result']
        
        # 게임 진행을 통해 결과를 얻어옴
        for move in item['history']:
            board_state = move['boardState']
            numeric_board = board_to_numeric(board_state)  # 보드 상태를 숫자 형태로 변환
            
            # 승리한 플레이어가 두었던 자리를 타겟으로 설정
            if move['player'] == result:
                y.append(move['row'] * 3 + move['col'])  # 0~8 범위로 위치를 변환
                X.append(numeric_board)  # 보드 상태를 입력으로 추가
                
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# 학습용 데이터 생성
X, y = generate_data(data)


# 학습용 데이터를 파일로 저장
np.save('train_x.npy', X)
np.save('train_y.npy', y)