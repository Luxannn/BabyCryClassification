import librosa

class AudioReader:


    @staticmethod
    def load(path, sr=None):
        # sr=None = keep original sample rate
        y, sr = librosa.load(path, sr=sr)
        return y, sr
