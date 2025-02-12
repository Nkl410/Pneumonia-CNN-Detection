# Pneumonia-CNN-Detection

### **Description du modèle**

Le modèle utilisé est un réseau de neurones convolutifs (CNN) conçu pour traiter des images de radiographies pulmonaires et détecter la présence de pneumonie. Il suit une architecture typique des CNN pour l'extraction des caractéristiques et la classification binaire.

### **Architecture du modèle**

Couche d'entrée : Images de 150x150 pixels en RGB.

**Convolution & Pooling :**

4 couches de Conv2D avec activation ReLU et padding 'same'.

4 couches de MaxPooling2D pour réduire la dimension.

**Aplatissement & Classification :**

Une couche Flatten() pour transformer les matrices en vecteurs.

Une couche Dense(256, activation='relu') pour la classification.

Une couche Dropout(0.5) pour éviter le surapprentissage.

Une couche Dense(1, activation='sigmoid') pour la sortie binaire.

```
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### **Optimisation et entraînement**

Fonction de perte : binary_crossentropy (classification binaire)

Optimiseur : adam

Métriques : accuracy

Techniques d'amélioration :

EarlyStopping pour arrêter l'entraînement si la validation ne s'améliore plus.

ReduceLROnPlateau pour réduire le taux d’apprentissage si la validation stagne.

```
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

history = model.fit(
    train_generator,
    epochs=20,  # Augmentation du nombre d'époques
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)
```


### **Résultats du modèle**

**Matrice de confusion**

Le modèle a été évalué à l'aide d'une matrice de confusion pour mesurer la performance de classification :

```
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
plt.xlabel('Prédiction')
plt.ylabel('Vérité')
plt.title('Matrice de Confusion')
plt.show()
```

### **Courbes de précision et de perte**

```
plt.plot(history.history['accuracy'], label='Précision - Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision - Validation')
plt.title('Précision du modèle')
plt.legend()
plt.show()
```
La courbe de perte montre une diminution progressive, indiquant que le modèle apprend correctement. Une stabilisation vers la fin montre que l’entraînement est bien optimisé.

```
plt.plot(history.history['loss'], label='Perte - Entraînement')
plt.plot(history.history['val_loss'], label='Perte - Validation')
plt.title('Perte du modèle')
plt.legend()
plt.show()
```
La courbe de précision affiche une amélioration rapide, démontrant que le modèle distingue efficacement les classes après quelques époques d'entraînement.


### **Exemple d’utilisation**

Tester une image de radiographie pour voir si elle est classée comme Pneumonia / Normal :

```
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('code/models/pneumonia_cnn.h5')
img_path = 'data/test/NORMAL/sample_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
result = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'
print(f'Prédiction : {result}')
```



