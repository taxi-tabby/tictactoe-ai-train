import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# Numpy 파일 로드
train_x = np.load('train_x3.npy')  # 보드 상태 (3x3 행렬)
train_y = np.load('train_y3.npy')  # 최적의 수 (One-Hot 가능성 있음)

# 🔥 해결 방법: 입력 데이터를 1D 벡터로 변환
train_x = train_x.reshape(-1, 9)  

# 🔥 해결 방법: train_y가 One-Hot Encoding이면 정수 레이블로 변환
if train_y.ndim > 1:  
    train_y = np.argmax(train_y, axis=1)  

# 🔥 해결 방법: train_y를 정수형(int)으로 변환
train_y = train_y.astype(np.int32)  


print(tf.__version__)
print('---------------------')
print(np.unique(train_x)) 
print('---------------------')
print(np.unique(train_x.shape))
print('---------------------')
print(np.unique(train_y)) 
print('---------------------')



# 모델 정의
def create_model():
    model = Sequential([
        Dense(512, input_dim=9, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dense(9, activation='softmax')
    ])


    # 모멘텀 추가하여 빠른 수렴 유도
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    return model

# 모델 생성
model = create_model()

# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,  
    restore_best_weights=True,
    verbose=1
)

# 모델 학습
model.fit(train_x, train_y, epochs=999, batch_size=32, 
          validation_split=0.2,  
          callbacks=[early_stopping])

# 모델 평가
loss, accuracy = model.evaluate(train_x, train_y)
print(f"✅ Model loss: {loss}, accuracy: {accuracy}")

# 모델 저장
model.save('tictactoe_model.h5')
print("🎯 Model saved as 'tictactoe_model.h5'")
