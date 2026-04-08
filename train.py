from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    # 1) Créer un dataset artificiel
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    feature_names = ["age", "salary", "score", "history"]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # 2) Séparer train / test
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    # 3) Configurer MLflow
    mlflow.set_experiment("mini_ml_project")

    with mlflow.start_run():
        # 4) Paramètres du modèle
        model = LogisticRegression(max_iter=300)

        # 5) Entraînement
        model.fit(X_train, y_train)

        # 6) Prédictions
        y_pred = model.predict(X_test)

        # 7) Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 8) Log MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("n_samples", 200)
        mlflow.log_param("n_features", 4)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        # 9) Sauvegarde locale du modèle
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)

        print("Entraînement terminé.")
        print(f"Accuracy : {accuracy:.3f}")
        print(f"F1-score : {f1:.3f}")
        print(f"Modèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    main()