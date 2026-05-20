import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_split import train_test_split # Optionnel mais recommandé pour scinder le dataset

# ==============================================================================
# VARIABLES IMPORTANTES (À CONFIGURER)
# ==============================================================================
CLEAN_DIR = "/home/bbapt/cesi/dataset/livrable2/source" # Dossier des images propres
NOISE_DIR = "/home/bbapt/cesi/dataset/livrable2/noise"  # Dossier des images bruitées

IMG_SIZE = 128                 # Taille de redimensionnement (ex: 128x128 ou 256x256)
BATCH_SIZE = 16                # Nombre d'images traitées à la fois
EPOCHS = 50                    # Nombre d'aller-retours d'entraînement
MODEL_SAVE_PATH = "blur_1_model.h5"
# ==============================================================================

def load_dataset():
    """Charge les paires d'images (bruitées et propres) en mémoire."""
    X_noise = []
    Y_clean = []
    
    print("Chargement et alignement du dataset...")
    
    # On parcourt le dossier de bruit pour retrouver l'image propre correspondante
    for noisy_name in os.listdir(NOISE_DIR):
        if "_noise_" in noisy_name:
            # Reconstitution du nom d'origine (ex: "img_1_noise_20.png" -> "img_1.png")
            base_name, rest = noisy_name.split("_noise_")
            ext = "." + rest.split(".")[-1]
            clean_name = base_name + ext
            
            clean_path = os.path.join(CLEAN_DIR, clean_name)
            noisy_path = os.path.join(NOISE_DIR, noisy_name)
            
            if os.path.exists(clean_path):
                # Lecture en couleur et redimensionnement
                img_clean = cv2.imread(clean_path)
                img_noisy = cv2.imread(noisy_path)
                
                img_clean = cv2.resize(img_clean, (IMG_SIZE, IMG_SIZE))
                img_noisy = cv2.resize(img_noisy, (IMG_SIZE, IMG_SIZE))
                
                # Normalisation des pixels entre 0 et 1 (crucial pour les réseaux de neurones)
                X_noise.append(img_noisy / 255.0)
                Y_clean.append(img_clean / 255.0)
                
    return np.array(X_noise), np.array(Y_clean)

def build_autoencoder(input_shape):
    """Définit l'architecture du réseau de neurones."""
    input_img = layers.Input(shape=input_shape)
    
    # ------------------------------------
    # ENCODER (Compression)
    # ------------------------------------
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # ------------------------------------
    # DECODER (Reconstruction)
    # ------------------------------------
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Couche de sortie : 3 canaux (RGB), activation 'sigmoid' pour sortir entre 0 et 1
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    return autoencoder

if __name__ == "__main__":
    # 1. Chargement des données
    X, Y = load_dataset()
    if len(X) == 0:
        print("Erreur : Aucune paire d'images correspondante trouvée. Vérifie tes dossiers.")
        exit()
        
    print(f"Dataset chargé : {len(X)} images bruitées prêtes.")
    
    # Séparation en Train (80%) et Validation (20%) pour surveiller l'overfitting
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 2. Création du modèle
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_autoencoder(input_shape)
    
    # Compilation : on utilise l'erreur quadratique moyenne (MSE) pour comparer pixel par pixel
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    # 3. Entraînement
    print("\nLancement de l'entraînement...")
    history = model.fit(
        X_train, Y_train, # Entrée : bruitée, Cible : propre
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val, Y_val)
    )
    
    # 4. Sauvegarde
    model.save(MODEL_SAVE_PATH)
    print(f"\nModèle entraîné et sauvegardé sous '{MODEL_SAVE_PATH}' !")