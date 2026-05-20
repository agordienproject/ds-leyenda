import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# ====================== CONFIGURATION ======================
print("TensorFlow version:", tf.__version__)
print("GPU disponible :", tf.config.list_physical_devices('GPU'))

# ====================== CHARGEMENT DES DONNÉES ======================
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalisation (0-1)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

print(f"Shape des images d'entraînement : {train_images.shape}")
print(f"Nombre de classes : {len(tf.unique(train_labels.flatten())[0])}")

# ====================== CRÉATION DU MODÈLE CNN ======================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # 10 classes
])

model.summary()  # Pour voir l'architecture

# ====================== COMPILATION ======================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ====================== ENTRAÎNEMENT ======================
epochs = 10  # Tu peux mettre 5 pour un test encore plus rapide

history = model.fit(train_images, train_labels, 
                    epochs=epochs,
                    batch_size=64,
                    validation_data=(test_images, test_labels),
                    verbose=1)

# ====================== ÉVALUATION FINALE ======================
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nPrécision sur le set de test : {test_acc:.4f} ({test_acc*100:.2f}%)")

# ====================== AFFICHAGE DES COURBES ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy Entraînement')
plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
plt.title('Précision')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Entraînement')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ====================== SAUVEGARDE DU MODÈLE ======================
# Format recommandé (.keras)
model.save('cifar10_model.keras')
print("Modèle sauvegardé sous : cifar10_model.keras")

# Optionnel : aussi en format SavedModel (dossier)
model.save('cifar10_model_saved')
print("Modèle également sauvegardé au format SavedModel dans le dossier 'cifar10_model_saved'")