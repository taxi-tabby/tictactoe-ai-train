import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Numpy 파일 로드
train_x = np.load('train_x3.npy')  # 보드 상태 (9개의 요소로 이루어진 벡터)
train_y = np.load('train_y3.npy')  # 최적의 수 (0~8의 값)


# print('---------------------')
# print(np.unique(train_x)) 
# print('---------------------')
# print(np.unique(train_x.shape))
# print('---------------------')
# print(np.unique(train_y)) 
# print('---------------------')

# 학습률 감소 콜백 (val_loss 기준)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# 모델 정의
def create_model():
    model = Sequential()
    model.add(Dense(256, input_dim=9, activation='relu'))
    model.add(BatchNormalization())  # 배치 정규화 추가
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='softmax'))  # 최종 출력층 (0~8의 위치 예측)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성
model = create_model()

# EarlyStopping 콜백 정의 (val_loss 기준)
early_stopping = EarlyStopping(
    monitor='val_loss',  # 검증 손실 기준
    patience=5,          # 5번 연속 개선되지 않으면 중단
    restore_best_weights=True,
    verbose=1
)

# 모델 학습 (검증 데이터 포함)
model.fit(train_x, train_y, epochs=999, batch_size=32, 
          validation_split=0.2,  # 검증 데이터 비율 추가
          callbacks=[lr_scheduler, early_stopping])

# 모델 평가
loss, accuracy = model.evaluate(train_x, train_y)
print(f"✅ Model loss: {loss}, accuracy: {accuracy}")

# 모델 저장
model.save('tictactoe_model.h5')
print("🎯 Model saved as 'tictactoe_model.h5'")
