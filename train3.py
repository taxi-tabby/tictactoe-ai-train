import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

# ✅ 1. Numpy 파일 로드
train_x = np.load('train_x_cnn.npy')  # 보드 상태 (3x3 행렬)
train_y = np.load('train_y_cnn.npy')  # 최적의 수 (정수 레이블)

# ✅ 2. CNN 입력 형식에 맞게 reshape
train_x = train_x.astype('float32').reshape(-1, 3, 3, 1)  # CNN 입력 형태로 변환

# ✅ 3. train_y가 One-Hot이면 정수 레이블로 변환
if train_y.ndim > 1:  
    train_y = np.argmax(train_y, axis=1)  

# ✅ 4. train_y를 정수형(int)으로 변환
train_y = train_y.astype(np.int32)  

print("🔹 TensorFlow Version:", tf.__version__)
print("🔹 train_x shape:", train_x.shape)  # (샘플 수, 3, 3, 1)
print("🔹 train_y shape:", train_y.shape)  # (샘플 수,)
print("🔹 Unique y labels:", np.unique(train_y))  # 0~8인지 확인

# ✅ 5. CNN 모델 정의 (패딩 추가 & 커널 크기 수정)
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(3,3,1), padding='same'),  
        Conv2D(64, (1,1), activation='relu'),  # ✅ 2x2 → 1x1로 변경
        Flatten(),  
        Dense(64, activation='relu'),
        Dropout(0.2),  
        Dense(32, activation='relu'),
        Dense(9, activation='softmax')  
    ])

    # 🔹 SGD + Momentum 사용
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ✅ 6. 모델 생성
model = create_model()

# ✅ 7. EarlyStopping 설정
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5,  
    restore_best_weights=True,
    verbose=1
)

# ✅ 8. 모델 학습
model.fit(train_x, train_y, epochs=999, batch_size=32, 
          validation_split=0.2,  
          callbacks=[early_stopping])

# ✅ 9. 모델 평가
loss, accuracy = model.evaluate(train_x, train_y)
print(f"✅ Model loss: {loss}, accuracy: {accuracy}")

# ✅ 10. 모델 저장
model.save('tictactoe_model.h5')
print("🎯 Model saved as 'tictactoe_model.h5'")
