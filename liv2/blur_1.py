import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split  # Correction de l'import ici

# ==============================================================================
# VARIABLES IMPORTANTES (À CONFIGURER)
# ==============================================================================
CLEAN_DIR = "/home/bbapt/cesi/dataset/livrable2/source" # Dossier des images propres
NOISE_DIR = "/home/bbapt/cesi/dataset/livrable2/noise"  # Dossier des images bruitées

IMG_SIZE = 128                 # Taille de redimensionnement (ex: 128x128)
BATCH_SIZE = 16                # Nombre d'images traitées à la fois
EPOCHS = 50                    # Nombre d'aller-retours d'entraînement

DIR_SAVE = "model/blur_1"                                       # Dossier de sauvegarde des modèles et historiques
MODEL_SAVE_PATH = f"{DIR_SAVE}/best_model_blur_1.h5"            # Sauvegardera UNIQUEMENT le meilleur
CSV_HISTORY_PATH = f"{DIR_SAVE}/training_history_blur_1.csv"    # Fichier d'historique des épochs
# ==============================================================================

def load_dataset():
    """Charge les paires d'images (bruitées et propres) en mémoire."""
    X_noise = []
    Y_clean = []
    
    print("Chargement et alignement du dataset...")
    
    if not os.path.exists(NOISE_DIR):
        print(f"Erreur : Le dossier {NOISE_DIR} n'existe pas.")
        return [], []

    for noisy_name in os.listdir(NOISE_DIR):
        if "_noise_" in noisy_name:
            base_name, rest = noisy_name.split("_noise_")
            ext = "." + rest.split(".")[-1]
            clean_name = base_name + ext
            
            clean_path = os.path.join(CLEAN_DIR, clean_name)
            noisy_path = os.path.join(NOISE_DIR, noisy_name)
            
            if os.path.exists(clean_path):
                img_clean = cv2.imread(clean_path)
                img_noisy = cv2.imread(noisy_path)
                
                img_clean = cv2.resize(img_clean, (IMG_SIZE, IMG_SIZE))
                img_noisy = cv2.resize(img_noisy, (IMG_SIZE, IMG_SIZE))
                
                X_noise.append(img_noisy / 255.0)
                Y_clean.append(img_clean / 255.0)
                
    return np.array(X_noise), np.array(Y_clean)

def build_autoencoder(input_shape):
    """Définit l'architecture de l'autoencodeur."""
    input_img = layers.Input(shape=input_shape)
    
    # ENCODER
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # DECODER
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    return autoencoder

if __name__ == "__main__":
    X, Y = load_dataset()
    if len(X) == 0:
        print("Erreur : Aucune paire d'images trouvée.")
        exit()
        
    print(f"Dataset chargé : {len(X)} images prêtes.")
    
    # Séparation Train (80%) / Validation (20%)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_autoencoder(input_shape)
    
    # Compilation avec la loss (MSE) et la métrique lisible (MAE)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()
    
    os.makedirs(DIR_SAVE, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # CONFIGURATION DES CALLBACKS
    # --------------------------------------------------------------------------
    callbacks = [
        # 1. Sauvegarde uniquement le meilleur modèle basé sur la val_loss minimale
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # 2. Sauvegarde automatique de l'historique dans un fichier CSV
        tf.keras.callbacks.CSVLogger(
            filename=CSV_HISTORY_PATH,
            separator=',',
            append=False
        )
    ]
    # --------------------------------------------------------------------------
    
    print("\nLancement de l'entraînement...")
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val, Y_val),
        callbacks=callbacks  # On applique nos outils ici
    )
    
    print(f"\nEntraînement terminé !")
    print(f"- Le meilleur modèle a été préservé dans : '{MODEL_SAVE_PATH}'")
    print(f"- L'historique complet a été écrit dans : '{CSV_HISTORY_PATH}'")