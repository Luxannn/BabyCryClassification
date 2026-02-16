import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA

class DatasetVisualizer:
   

    @staticmethod
    def class_distribution(csv_file):
        data = pd.read_csv(csv_file)
        counts = data["label"].value_counts()

        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Class Distribution")
        plt.xlabel("Cry Category")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

        return counts

    @staticmethod
    def pca_scatter(csv_file, sample_size=500):
        data = pd.read_csv(csv_file)
        X = data.drop(columns=["label"])
        y = data["label"]

        # Replace inf/NaN and drop bad rows
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]

        # Downsample for speed
        if len(X) > sample_size:
            sampled = X.sample(sample_size, random_state=42)
            y = y.loc[sampled.index]
            X = sampled

        if X.shape[1] < 2:
            raise ValueError("Not enough valid features for PCA.")

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        df_pca = pd.DataFrame({
            "Feature Mix 1": X_pca[:, 0],
            "Feature Mix 2": X_pca[:, 1],
            "label": y.values
        })

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df_pca, x="Feature Mix 1", y="Feature Mix 2", hue="label",
            palette="Set1", alpha=0.7, s=60
        )
        plt.title("Cry Samples Projected into 2D Feature Space")
        plt.legend(title="Category")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def average_feature_bar(csv_file, n_features=5):
        data = pd.read_csv(csv_file)
        features = data.drop(columns=["label"])
        labels = data["label"]

        # Compute mean of first N features per class
        avg = features.iloc[:, :n_features].join(labels).groupby("label").mean()

        avg.T.plot(kind="bar", figsize=(10, 6))
        plt.title(f"Average of First {n_features} Features per Class")
        plt.xlabel("Feature Index")
        plt.ylabel("Average Value")
        plt.legend(title="Cry Category")
        plt.tight_layout()
        plt.show()
