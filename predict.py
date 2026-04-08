import joblib
import pandas as pd

# 1. Charger le modèle
model = joblib.load("model/model.joblib")

# 2. Créer une donnée AVEC noms de colonnes
new_data = pd.DataFrame([{
    "age": 30,
    "salary": 50000,
    "score": 0.8,
    "history": 1
}])

# 3. Prédiction
prediction = model.predict(new_data)

print("Prédiction :", prediction[0])