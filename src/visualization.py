import matplotlib.pyplot as plt
import librosa.display

class Visualizer:
   

    @staticmethod
    def waveform(y, sr, title="Waveform"):
        plt.figure(figsize=(10, 3))
        plt.title(title)
        librosa.display.waveshow(y, sr=sr)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    @staticmethod
    def spectrogram(y, sr, title="Spectrogram"):
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(10, 4))
        plt.title(title)
        librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
        plt.colorbar(format="%+2.0f dB")
        plt.show()
