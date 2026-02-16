import os
import pandas as pd
from src.audio_reader import AudioReader
from src.feature_extraction import FeatureExtractor

class DatasetBuilder:


    @staticmethod
    def build_dataset(output_csv, *dataset_dirs):
        features = []
        labels = []

        for data_dir in dataset_dirs:
            # Each dataset is expected to have subfolders = class labels
            for category in os.listdir(data_dir):
                category_path = os.path.join(data_dir, category)

                if not os.path.isdir(category_path):
                    continue

                for filename in os.listdir(category_path):
                    if not filename.endswith(".wav"):
                        continue

                    file_path = os.path.join(category_path, filename)

                    try:
                        # Load audio
                        y, sr = AudioReader.load(file_path)

                        # Extract features
                        feats = FeatureExtractor.extract_all(y, sr)

                        features.append(feats)
                        labels.append(category)

                    except Exception as e:
                        print(f"Ошибка {file_path}: {e}")
                        continue

        # Save dataset
        data = pd.DataFrame(features)
        data["label"] = labels
        data.to_csv(output_csv, index=False)
        print(f"✅ Dataset saved to {output_csv} with {len(data)} samples.")
