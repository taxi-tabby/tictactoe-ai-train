import numpy as np
import tensorflow as tf
import os
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm  
from utils import create_model1, dynamicBatchSize

# EarlyStopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)





# ✅ 폴더 내의 파일들을 확인하고, 필요한 데이터를 로딩
directory = './npy'
files = [f for f in os.listdir(directory) if f.endswith('.npy')]

# 패턴을 이용해 'train_x'와 'train_y' 파일을 매칭
pattern = re.compile(r"train_(x|y)_(\d+)x(\d+)\.npy")

train_files = {'x': {}, 'y': {}}

# 파일 이름에서 x, y 데이터 분리하여 dictionary에 저장
for file in files:
    match = pattern.match(file)
    if match:
        file_type = match.group(1)  # 'x' 또는 'y'
        x_size = int(match.group(2))  # x 크기
        y_size = int(match.group(3))  # y 크기
        shape = (x_size, y_size)
        
        # x와 y 파일을 구분하여 저장
        if file_type == 'x':
            train_files['x'][shape] = file
        elif file_type == 'y':
            train_files['y'][shape] = file

dataLengthCollector = []

# ✅ 배치 크기를 미리 구하기 위해 먼저 접근해서 사이즈를 저장
for shape in tqdm(train_files['x'].keys(), desc="Collecting Data Lengths all training data"):
    x_file = train_files['x'].get(shape)
    y_file = train_files['y'].get(shape)
    
    train_x = np.load(os.path.join(directory, x_file))
    data_size = train_x.shape[0]
    
    dataLengthCollector.append(data_size)



print(f"🔹 ----------------------------------------------------------------------------------")
print(f"🔹 training size: {dataLengthCollector}:")


# ✅ 모델 학습을 위한 루프
for shape in tqdm(train_files['x'].keys(), desc="Training Models"):

    x_file = train_files['x'].get(shape)
    y_file = train_files['y'].get(shape)

    # x와 y 파일이 모두 존재하는 경우에만 진행
    if x_file and y_file:
        
        # ✅ Numpy 파일 로드
        train_x = np.load(os.path.join(directory, x_file))
        train_y = np.load(os.path.join(directory, y_file))

        # ✅ 동적으로 보드 크기 추출
        data_size, x_size, y_size = train_x.shape[0], train_x.shape[1], train_x.shape[2]  # x_size, y_size를 동적으로 추출

        batch_size = dynamicBatchSize(dataLengthCollector, data_size)

        train_x_shape = np.expand_dims(train_x, axis=-1)         
        train_y_int = np.argmax(train_y, axis=1)


        # ✅ CNN 입력 형식에 맞게 reshape
        # train_x = train_x.astype('float32').reshape(-1, shape[0], shape[1], 1)  # CNN 입력 형태로 변환
        # train_x와 train_y는 이미 One-Hot 인코딩 되어 있음. 추가 변환 필요 없음.

        # ✅ train_x와 train_y의 길이가 일치하는지 확인
        if train_x.shape[0] != train_y.shape[0]:
            print(f"❌ train_x and train_y lengths do not match for shape {shape}: {train_x.shape[0]} vs {train_y.shape[0]}")
            continue  # 이 경우 해당 크기 모델 학습을 건너뜁니다.

        print(f"🔹 ----------------------------------------------------------------------------------")
        print(f"🔹 Model Training for size {shape}:")
        print(f"   train_x shape: {train_x.shape}")  
        print(f"   train_y shape: {train_y.shape}")  
        print(f"   batch_size: {data_size} -> {batch_size}")

        # ✅ 모델 생성
        input_shape = train_x_shape.shape[1:]  # (height, width, channels)

        # 동적으로 모델 생성
        # model = create_improved_model(input_shape, train_y.shape[-1])  # num_classes는 train_y의 마지막 차원의 크기
        model = create_model1(input_shape) 
        
        # 모델 훈련
        model.fit(train_x_shape, train_y_int, epochs=2000, batch_size=batch_size, validation_split=0.01, callbacks=[reduce_lr, early_stopping])
        
        # 모델 평가
        loss, accuracy = model.evaluate(train_x_shape, train_y_int)
        print(f"✅ Model loss: {loss}, accuracy: {accuracy}")

        # 모델 저장
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_filename = f"tictactoe_model_{shape[0]}x{shape[1]}.h5"
        model.save(os.path.join(model_dir, model_filename))
        print(f"🎯 Model saved as '{model_filename}'")
    
    else:
        print(f"❌ Missing x or y data for shape {shape}. Skipping...")

print("🎉 All models have been trained and saved.")
