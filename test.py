import librosa

path = 'dataset/a_1.wav'
y1, sr1 = librosa.load(path, sr=16000, duration=2.97)
print(y1.shape)
ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
print(ps.shape)
