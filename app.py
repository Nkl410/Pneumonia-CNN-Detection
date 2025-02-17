from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model  # type: ignore

app = FastAPI()

# Charger le modèle sauvegardé
model = load_model("X_ray_cnn.h5")

# Recompiler le modèle pour éviter le warning
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Classes du modèle (modifie si nécessaire)
class_names = ['NORMAL', 'PNEUMONIA']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lire l'image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    

    # Prétraitement (adapté au modèle)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Prédiction
    prediction = model.predict(image)[0][0]
    label = class_names[int(prediction > 0.5)]

    return {"prediction": label, "confidence": float(prediction)}
