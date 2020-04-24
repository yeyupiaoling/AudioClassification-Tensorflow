import librosa
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('models/cnn.h5')


# 读取音频数据
def load_data(data_path):
    y1, sr1 = librosa.load(data_path, duration=2.97)
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    result = model.predict(data)
    lab = tf.argmax(result, 1)
    return lab


if __name__ == '__main__':
    # 要预测的音频文件
    path = ''
    label = infer(path)
    print('音频：%s 的预测结果标签为：%d' % (path, label))
