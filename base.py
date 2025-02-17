import tensorflow as tf
from keras import models
from keras import layers
from keras import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # type: ignore 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras import Input # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report # type: ignore
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore




# Générateur pour les données d'entraînement (avec augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # Normalisation des pixels
    rotation_range=20,        # Rotation aléatoire jusqu'à 20 degrés
    width_shift_range=0.2,    # Décalage horizontal
    height_shift_range=0.2,   # Décalage vertical
    shear_range=0.2,          # Cisaillement
    zoom_range=0.2,           # Zoom aléatoire
    horizontal_flip=True,     # Flip horizontal
    fill_mode='nearest'       # Remplissage des pixels manquants
)

# Générateur pour les données de test (pas d'augmentation ici)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Dossier d'entraînement
train_generator = train_datagen.flow_from_directory(
    '/Users/nikolavucic/Documents/ML/CNN/.venv/xray_dataset_covid19/train',           # Chemin vers le dossier train
    target_size=(150, 150),    # Redimensionner les images à 150x150 pixels
    batch_size=32,             # Nombre d'images chargées en mémoire par lot
    class_mode='binary'        # Type de sortie : binaire (0 ou 1)
)

# Dossier de test
test_generator = test_datagen.flow_from_directory(
    '/Users/nikolavucic/Documents/ML/CNN/.venv/xray_dataset_covid19/test',            # Chemin vers le dossier test
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print(train_generator.class_indices)


#3. Construction du Modèle CNN


model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

 # Nombre d'époques?? 
 # Nombre ???

# Entraîner le modèle
# Call back d'arrêt anticiper 
history = model.fit(
    train_generator,
    epochs=14,                      # Nombre d'époques
    validation_data=test_generator  # Données de validation pour surveiller les performances
)



#4. Compilation et Entraînement
#Une fois le modèle construit, il doit être compilé avec :

#Une fonction de perte adaptée : binary_crossentropy (classification binaire).
#Un optimiseur : adam est souvent un bon choix pour commencer.

loss, accuracy = model.evaluate(test_generator)
#print(f"Précision sur les données de test : {accuracy * 100:.2f}%")


# Courbes de précision

plt.plot(history.history['accuracy'], label='Précision - Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision - Validation')
plt.title('Précision du modèle')
plt.legend()
plt.show()


# Courbes de perte
plt.plot(history.history['loss'], label='Perte - Entraînement')
plt.plot(history.history['val_loss'], label='Perte - Validation')
plt.title('Perte du modèle')
plt.legend()
plt.show()

### **6. Matrice de Confusion**
#Une **matrice de confusion** montre où le modèle se trompe. Par exemple :
#- Combien de poumons **sains** sont mal classés comme **malades** ?
#- Combien de poumons **malades** sont mal classés comme **sains** ?

#python

# Prédictions
y_pred = (model.predict(test_generator) > 0.5).astype("int32")
y_true = test_generator.classes

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
print("Matrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))
