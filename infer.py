import librosa
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('models/resnet50.h5')


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    assert len(wav_output) >= 8000, "有效音频小于0.5s"
    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    result = model.predict(data)
    lab = tf.argmax(result, 1)
    return lab


if __name__ == '__main__':
    # 要预测的音频文件
    path = 'dataset/UrbanSound8K/audio/fold6/121285-0-0-3.wav'
    label = infer(path)
    print('音频：%s 的预测结果标签为：%d' % (path, label))
