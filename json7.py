import json
import numpy as np
import glob
import random
import os
from tqdm import tqdm  # ✅ 진행 상태 표시
from utils import board_to_numeric, value_to_numeric, longest_sequence # 유틸리티 함수 가져오기

# JSON 파일 경로 패턴
json_file_pattern = './data/data-*.json'

# JSON 파일 읽기
data = []
json_files = sorted(glob.glob(json_file_pattern))
random.shuffle(json_files)  # 파일 순서 셔플

for json_file_path in tqdm(json_files, desc="📂 JSON File Shuffle load ", unit="file"):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data.extend(json.load(file))
    except Exception as e:
        print(f"❌ 오류 발생: {json_file_path} - {e}")

if not data:
    print("\U0001F6A8 JSON 파일을 찾을 수 없습니다! 경로를 확인하세요.")
    exit()

# 🟢 데이터셋 생성 함수
def generate_data(data):
    # 학습 데이터 리스트
    X, y = [], []  
    # 보드 크기와 승리 조건별로 분리할 딕셔너리
    data_by_size_and_condition = {}

    valid_data_count = 0  # 유효한 데이터의 개수
    board_size_counts = {}  # 보드 크기별 데이터 개수 기록

    for item in tqdm(data, desc="🔄 Data transform", unit="Game"):
        
        rows = 0
        cols = 0
        num_channels = 0
        
        if item['history']:
            board = item['history'][0]['boardState']
            rows = len(board)
            cols = len(board[0]) if rows > 0 else 0
            num_channels = rows * cols 

        # 공간 없으면 저리 가쇼
        if num_channels == 0:
            continue
        
        # x : 1, o : -1, 빈칸 : 0
        if item['result'] not in ['X', 'O']:
            continue

        # 마지막 게임의 보드 상태 가져오기
        last_board_state = item['history'][-1]['boardState']
        last_numberic_board = board_to_numeric(last_board_state, rows, cols)
        winning_condition = longest_sequence(last_numberic_board)
        
        # 데이터 저장을 위한 임시 변수
        count = 0
        dataTuple = {"x": None, "y": None}
        
        for move in item['history']:
            picked = None
            countWillUp = False
            
            picked = move  # 모든 수를 지정
            
            if picked is not None:
                board = picked['boardState']
                
                if count % 2 == 0 and countWillUp is False:
                    dataTuple['x'] = board_to_numeric(board, rows, cols)
                    countWillUp = True
                
                if count % 2 == 1 and countWillUp is False:
                    dataTuple['y'] = board_to_numeric(board, rows, cols).flatten()
                    countWillUp = True
                
                if countWillUp:
                    count += 1
                    valid_data_count += 1
                
                if dataTuple['x'] is not None and dataTuple['y'] is not None:
                    X.append(dataTuple['x'])
                    y.append(dataTuple['y'])
                    dataTuple = {"x": None, "y": None}

        # 보드 크기별로 분리
        board_size = (rows, cols)
        board_size_counts[board_size] = board_size_counts.get(board_size, 0) + 1

        # winning_condition과 board_size별로 데이터를 분류
        key = (board_size, winning_condition)
        if key not in data_by_size_and_condition:
            data_by_size_and_condition[key] = {"X": [], "y": []}

        data_by_size_and_condition[key]["X"].extend(X)
        data_by_size_and_condition[key]["y"].extend(y)

    # 보드 크기와 승리 조건별로 데이터를 저장
    output_dir = './npy/'
    os.makedirs(output_dir, exist_ok=True)

    for (board_size, winning_condition), datasets in data_by_size_and_condition.items():
        X_data = np.array(datasets["X"], dtype=np.float32)
        y_data = np.array(datasets["y"], dtype=np.float32)

        # 보드 크기와 승리 조건을 모두 고려한 파일 이름 저장
        np.save(f'{output_dir}train_x_{board_size[0]}x{board_size[1]}_{winning_condition}.npy', X_data)
        np.save(f'{output_dir}train_y_{board_size[0]}x{board_size[1]}_{winning_condition}.npy', y_data)
        print(f"🎯 {board_size[0]}x{board_size[1]} 보드 크기, 승리 조건 {winning_condition}에 맞춰 학습 데이터 저장 완료: train_x_{board_size[0]}x{board_size[1]}_{winning_condition}.npy, train_y_{board_size[0]}x{board_size[1]}_{winning_condition}.npy")

    print(f"✅ 변환된 데이터 개수: {valid_data_count}")  # 최종적으로 추가된 유효한 데이터 개수 출력
    
    return data_by_size_and_condition


# 🟢 데이터 생성 및 저장
if __name__ == '__main__':
    data_by_size_and_condition = generate_data(data)  # 데이터 생성 함수 호출 및 반환값을 data_by_size_and_condition에 할당