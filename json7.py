import json
import numpy as np
import glob
import random
import os
from tqdm import tqdm  # ✅ 진행 상태 표시
from utils import board_to_numeric, value_to_numeric, longest_sequence  # 유틸리티 함수 가져오기

# JSON 파일 경로 패턴
json_file_pattern = './data/data-*.json'

# JSON 파일 읽기
data = []
json_files = sorted(glob.glob(json_file_pattern))
random.shuffle(json_files)  # 파일 순서 셔플

# 모든 JSON 파일을 읽어서 `data`에 누적
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
    # 보드 크기와 승리 조건을 기준으로 조건 목록 만들기
    conditions_set = set()
    for item in data:
        if 'history' in item and item['history']:  # history가 존재하고 비어있지 않은 경우만 처리
            board = item['history'][0]['boardState']
            rows = len(board)
            cols = len(board[0]) if rows > 0 else 0
            last_board_state = item['history'][-1]['boardState']
            last_numberic_board = board_to_numeric(last_board_state, rows, cols)
            winning_condition = longest_sequence(last_numberic_board)
            conditions_set.add((rows, cols, winning_condition))  # 보드 크기 + 승리 조건 튜플 생성

    # 조건별로 데이터를 누적하여 저장
    valid_data_count = 0  # 유효한 데이터의 개수
    output_dir = './npy/'
    os.makedirs(output_dir, exist_ok=True)

    # 조건별로 데이터 누적
    for condition in tqdm(conditions_set, desc="🔄 Processing conditions", unit="condition"):
        rows, cols, winning_condition = condition  # 3개의 값으로 unpack 수정
        X_data, y_data = [], []

        # 데이터셋을 조건에 맞게 순차적으로 처리
        for item in data:
            if 'history' not in item or not item['history']:  # history가 없거나 빈 리스트인 경우 건너뛰기
                continue

            # 현재 조건에 맞는 보드 크기와 승리 조건만 처리
            current_rows = len(item['history'][0]['boardState'])
            current_cols = len(item['history'][0]['boardState'][0])

            # 게임 결과가 'X' 또는 'O'여야만 유효
            if item['result'] not in ['X', 'O']:
                continue

            # 마지막 보드 상태와 승리 조건
            last_board_state = item['history'][-1]['boardState']
            last_numberic_board = board_to_numeric(last_board_state, current_rows, current_cols)
            current_winning_condition = longest_sequence(last_numberic_board)

            if current_winning_condition != winning_condition or (current_rows, current_cols) != (rows, cols):
                continue  # 현재 조건에 맞지 않는 데이터는 건너뛰기

            # X와 y 데이터를 계산하여 저장
            count = 0
            dataTuple = {"x": None, "y": None}

            # 게임의 각 이동에 대해 처리
            for move in item['history']:
                board = move['boardState']

                if count % 2 == 0:  # X 플레이어
                    dataTuple['x'] = board_to_numeric(board, current_rows, current_cols)
                elif count % 2 == 1:  # O 플레이어
                    dataTuple['y'] = board_to_numeric(board, current_rows, current_cols).flatten()

                if dataTuple['x'] is not None and dataTuple['y'] is not None:
                    X_data.append(dataTuple['x'])  # X 데이터 누적
                    y_data.append(dataTuple['y'])  # y 데이터 누적
                    dataTuple = {"x": None, "y": None}
                    valid_data_count += 1

                count += 1

        # 조건에 맞는 데이터가 있으면 한번에 저장
        if X_data and y_data:
            X_data = np.array(X_data, dtype=np.float32)
            y_data = np.array(y_data, dtype=np.float32)

            # 파일 저장
            np.save(f'{output_dir}train_x_{rows}x{cols}_{winning_condition}.npy', X_data)
            np.save(f'{output_dir}train_y_{rows}x{cols}_{winning_condition}.npy', y_data)
            # print(f"🎯 {rows}x{cols} 보드 크기, 승리 조건 {winning_condition}에 맞춰 학습 데이터 저장 완료")

    print(f"✅ 변환된 데이터 개수: {valid_data_count}")

# 🟢 데이터 생성 및 저장
if __name__ == '__main__':
    generate_data(data)  # 데이터 생성 함수 호출
