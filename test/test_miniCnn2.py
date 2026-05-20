import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time

# ====================== CONFIGURATION ======================
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU disponible :", gpus)

# ====================== CHARGEMENT DES DONNÉES ======================
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images.astype('float32') / 255.0
test_images  = test_images.astype('float32') / 255.0

# ====================== CRÉATION DU MODÈLE ======================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.summary()

# ====================== COMPILATION ======================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# ====================== ENTRAÎNEMENT AVEC CHRONOMÈTRE ======================
epochs = 10

print(f"\nDébut de l'entraînement sur {epochs} epochs...\n")
start_time = time.time()                     # ← Début du chrono

history = model.fit(train_images, train_labels,
                    epochs=epochs,
                    batch_size=64,
                    validation_data=(test_images, test_labels),
                    verbose=1)

end_time = time.time()                       # ← Fin du chrono

# ====================== CALCUL ET AFFICHAGE DU TEMPS ======================
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print("\n" + "="*50)
print("✅ ENTRAÎNEMENT TERMINÉ")
print(f"Temps total d'entraînement : {total_seconds:.1f} secondes")
print(f"                      soit : {minutes} minute(s) et {seconds} seconde(s)")
print("="*50)

# Temps par epoch (approximatif)
avg_time_per_epoch = total_seconds / epochs
print(f"Temps moyen par epoch : {avg_time_per_epoch:.1f} secondes")

# ====================== ÉVALUATION ======================
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Précision sur le set de test : {test_acc:.4f} ({test_acc*100:.2f}%)")

# ====================== AFFICHAGE DES COURBES ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entraînement')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entraînement')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ====================== SAUVEGARDE ======================
model.save('cifar10_model.keras')
print("\nModèle sauvegardé sous : cifar10_model.keras")