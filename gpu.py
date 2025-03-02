import tensorflow as tf

# GPU가 있는지 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU가 {len(gpus)}개 발견되었습니다.")
    try:
        # 첫 번째 GPU만 사용하도록 설정
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # GPU 메모리 제한을 두고 사용 (예: 4GB만 사용)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
        print("GPU 메모리 할당 설정 완료")
    except RuntimeError as e:
        print(e)
else:
    print("GPU가 발견되지 않았습니다.")
