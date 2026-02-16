import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate(csv_file):
    # Load dataset
    data = pd.read_csv(csv_file)

    # Split features and labels
    X = data.drop(columns=["label"])
    y = data["label"]

    # Replace any inf/nan values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Model: Random Forest (simple & interpretable)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}\n")

    # Detailed report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes_, yticklabels=model.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix of Baby Cry Classifier")
    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    train_and_evaluate("child_cry_data_combined.csv")
