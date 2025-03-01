import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Numpy 파일 로드
train_x = np.load('train_x.npy')  # 보드 상태 (9개의 요소로 이루어진 벡터)
train_y = np.load('train_y.npy')  # 최적의 수 (0~8의 값)

# 모델 정의
def create_model():
    model = Sequential()
    model.add(Dense(64, input_dim=9, activation='relu'))  # 입력 차원은 9 (보드 상태)
    model.add(Dense(32, activation='relu'))  # 두 번째 은닉층
    model.add(Dense(9, activation='softmax'))  # 9개의 위치를 예측 (0~8)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성
model = create_model()

# 모델 학습
model.fit(train_x, train_y, epochs=50, batch_size=32)

# 모델 평가
loss, accuracy = model.evaluate(train_x, train_y)
print(f"Model loss: {loss}, accuracy: {accuracy}")

# 모델 저장
model.save('tictactoe_model.h5')
print("Model saved as 'tictactoe_model.h5'")