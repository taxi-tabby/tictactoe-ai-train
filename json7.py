import json
import numpy as np
import glob
import random
import os
from tqdm import tqdm  # ✅ 진행 상태 표시
from utils import board_to_numeric, value_to_numeric # 유틸리티 함수 가져오기

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
    X, y = [], []  # 학습 데이터 저장 리스트

    board_sizes = set()  # 보드 크기별로 데이터 구분할 때 사용
    valid_data_count = 0  # 실제로 추가된 유효한 데이터의 개수
    board_size_counts = {}  # 보드 크기별로 데이터 개수 기록

    for item in tqdm(data, desc="🔄 Data transform", unit="Game"):
        
        rows = 0
        cols = 0
        num_channels = 0
        
        # print(f"item : {item["history"]}")
        
        if item['history']:
            board = item['history'][0]['boardState']
            rows = len(board)
            cols = len(board[0]) if rows > 0 else 0
            num_channels = rows * cols 

            # board_state = move['boardState']
            # rows = len(board_state)
            # cols = len(board_state[0]) if rows > 0 else 0


        # 공간 없으면 저리 가쇼
        if num_channels == 0:
            print(f"No board channels found : {num_channels}")
            continue
        
        # x : 1, o : -1, 빈칸 : 0
        if item['result'] not in ['X', 'O']:
            continue

        # 승자 반환. 수집하기 위함.
        winner: int = value_to_numeric(item['result'])


        # 이전에 있던 패자의 수를 먼저 저장
        count = 0
        dataTuple = {"x": None, "y": None};
        for move in item['history']:
            picked = None
            if move['player'] == 'X' and winner == 1:  # 'X' 승리
                picked = move
            elif move['player'] == 'O' and winner == -1:  # 'O' 승리
                picked = move
                
            #학습용 데이터 입력(패자의 현재 보드 상태)
            if picked is not None:
                if count % 2 == 0:
                    dataTuple['x'] = board_to_numeric(picked['boardState'], rows, cols)
                    valid_data_count += 1
                    ++count
                
                if count % 2 == 1:
                    dataTuple['y'] = board_to_numeric(picked['boardState'], rows, cols).flatten()
                    valid_data_count += 1
                    ++count
                
                if dataTuple['x'] is not None and dataTuple['y'] is not None:
                    print(f"Data Tuple : {len(dataTuple['x'])} / {len(dataTuple['y'])}")
                    X.append(dataTuple[0])
                    y.append(dataTuple[1])
                    dataTuple = (None, None)
                    
            

    
        # 각 보드 크기별로 분리
        board_size = (rows, cols)
        board_sizes.add(board_size)



        # 보드 크기별 카운트 업데이트
        if board_size not in board_size_counts:
            board_size_counts[board_size] = 0
        board_size_counts[board_size] += 1
    

            
            



    # 보드 크기별로 데이터를 저장
    print(f"Board Size : {board_sizes}")
    data_by_size = {size: {"X": [], "y": []} for size in board_sizes}

    for x, y in zip(X, y):
        board_size = (x.shape[0], x.shape[1])
        data_by_size[board_size]["X"].append(x)
        data_by_size[board_size]["y"].append(y)

    print(f"Data by Size : {data_by_size}")

    # 각 보드 크기별로 데이터를 파일로 저장
    output_dir = './npy/'
    os.makedirs(output_dir, exist_ok=True)



    for size, datasets in data_by_size.items():
        X_data = np.array(datasets["X"], dtype=np.float32)
        y_data = np.array(datasets["y"], dtype=np.float32)
        np.save(f'{output_dir}train_x_{size[0]}x{size[1]}.npy', X_data)
        np.save(f'{output_dir}train_y_{size[0]}x{size[1]}.npy', y_data)
        # print(f"🎯 {size[0]}x{size[1]} 보드 크기 학습 데이터 저장 완료: train_x_{size[0]}x{size[1]}.npy, train_y_{size[0]}x{size[1]}.npy")

    print(f"✅ 변환된 데이터 개수: {valid_data_count}")  # 최종적으로 추가된 유효한 데이터 개수 출력
    
    # 보드 크기별 데이터 카운트 출력
    # print("\n📊 각 보드 크기별 데이터 개수:")
    # for size, count in board_size_counts.items():
    #     print(f"{size[0]}x{size[1]} 보드 크기: {count}개 데이터")

    return data_by_size


# 🟢 데이터 생성 및 저장
if __name__ == '__main__':
    data_by_size = generate_data(data)  # 데이터 생성 함수 호출 및 반환값을 data_by_size에 할당
