import numpy as np
import librosa

class FeatureExtractor:

    @staticmethod
    def extract_all(y, sr, n_mfcc=20):
        feats = []

        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        feats.extend(np.mean(mfccs, axis=1))

        # Delta MFCC
        deltas = librosa.feature.delta(mfccs)
        feats.extend(np.mean(deltas, axis=1))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats.extend(np.mean(chroma, axis=1))

        # Spectral contrast
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            feats.extend(np.mean(contrast, axis=1))
        except Exception:
            feats.extend([0, 0, 0, 0, 0, 0])

        # Tonnetz
        try:
            tonnetz = librosa.feature.tonnetz(
                y=librosa.effects.harmonic(y), sr=sr
            )
            feats.extend(np.mean(tonnetz, axis=1))
        except Exception:
            feats.extend([0, 0, 0, 0, 0, 0])

        return np.array(feats)
