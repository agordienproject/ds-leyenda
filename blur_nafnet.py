import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==============================================================================
# VARIABLES IMPORTANTES (À CONFIGURER)
# ==============================================================================
CLEAN_DIR = "/home/bbapt/cesi/dataset/livrable2/source" # Dossier des images propres
NOISE_DIR = "/home/bbapt/cesi/dataset/livrable2/noise"  # Dossier des images bruitées

IMG_SIZE = 256                 # Taille de redimensionnement
BATCH_SIZE = 32                # Nombre d'images traitées à la fois
EPOCHS = 150                   # Nombre d'aller-retours d'entraînement

DIR_SAVE = "model/blur_nafnet"                                       # Dossier de sauvegarde spécifique
MODEL_SAVE_PATH = f"{DIR_SAVE}/best_model_blur_nafnet.h5"            # Sauvegardera UNIQUEMENT le meilleur
CSV_HISTORY_PATH = f"{DIR_SAVE}/training_history_blur_nafnet.csv"    # Fichier d'historique des épochs
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

# ==============================================================================
# COMPOSANTS ARCHITECTURAUX NAFNET
# ==============================================================================

def simple_gate(x):
    """Remplace l'activation non-linéaire. 
    Divise les canaux en deux et applique un produit élément par élément.
    """
    c1, c2 = tf.split(x, num_or_size_splits=2, axis=-1)
    return c1 * c2

def simplified_channel_attention(x):
    """Mécanisme d'attention par canal sans surcoût de calcul."""
    # Global Average Pooling spatiale pour condenser l'information (B, 1, 1, C)
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    # Linéarisation via convolution 1x1 pour ajuster les poids des canaux
    ca = layers.Conv2D(x.shape[-1], (1, 1), padding='same', use_bias=False)(gap)
    return x * ca

def naf_block(x, filters):
    """Bloc fondamental de NAFNet (LayerNorm -> Conv1x1 -> DWConv3x3 -> Gate -> SCA -> Conv1x1)."""
    shortcut = x
    
    # 1. Normalisation de couche
    x = layers.LayerNormalization(axis=-1)(x)
    
    # 2. Expansion (on multiplie les canaux par 2 car le SimpleGate va les diviser par 2)
    x = layers.Conv2D(filters * 2, (1, 1), padding='same', use_bias=False)(x)
    
    # 3. Convolution de profondeur (Depthwise) pour capturer l'espace environnant
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    
    # 4. Mécanisme de Gate (Zéro fonction d'activation de type ReLU/GELU ici)
    x = layers.Lambda(simple_gate)(x)
    
    # 5. Attention de canal simplifiée
    x = simplified_channel_attention(x)
    
    # 6. Projection de sortie pour retrouver l'épaisseur initiale
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    
    # Connexion résiduelle
    return layers.add([shortcut, x])

def build_nafnet(input_shape):
    """Construit une architecture de type NAFNet structurée en U."""
    input_img = layers.Input(shape=input_shape)
    
    # Projection initiale de l'image RGB (3 canaux -> 32 filtres)
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(input_img)
    
    # --- ENCODER (Descente) ---
    # Niveau 1 (256x256)
    x1 = naf_block(x, 32)
    # Downsampling par convolution de stride 2 (alternative performante au MaxPool)
    down1 = layers.Conv2D(64, (2, 2), strides=2, padding='same', use_bias=False)(x1) # -> 128x128
    
    # Niveau 2 (128x128)
    x2 = naf_block(down1, 64)
    down2 = layers.Conv2D(128, (2, 2), strides=2, padding='same', use_bias=False)(x2) # -> 64x64
    
    # --- BOTTLENECK (Le fond du U) ---
    bottleneck = naf_block(down2, 128)
    
    # --- DECODER (Remontée) ---
    # Remontée au Niveau 2 (128x128)
    up2 = layers.UpSampling2D((2, 2))(bottleneck)
    up2 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(up2)
    concat2 = layers.Concatenate()([up2, x2])
    dec2 = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(concat2)
    dec2 = naf_block(dec2, 64)
    
    # Remontée au Niveau 1 (256x256)
    up1 = layers.UpSampling2D((2, 2))(dec2)
    up1 = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(up1)
    concat1 = layers.Concatenate()([up1, x1])
    dec1 = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(concat1)
    dec1 = naf_block(dec1, 32)
    
    # --- COUCHE DE SORTIE ---
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(dec1)
    
    model = models.Model(input_img, decoded)
    return model

# ==============================================================================
# PIPELINE D'ENTRAÎNEMENT
# ==============================================================================

if __name__ == "__main__":
    X, Y = load_dataset()
    if len(X) == 0:
        print("Erreur : Aucune paire d'images trouvée.")
        exit()
        
    print(f"Dataset chargé : {len(X)} images prêtes.")
    
    # Séparation Train (80%) / Validation (20%)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    model = build_nafnet(input_shape)
    
    # Compilation
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.summary()
    
    os.makedirs(DIR_SAVE, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=CSV_HISTORY_PATH,
            separator=',',
            append=False
        )
    ]
    
    print("\nLancement de l'entraînement NAFNet...")
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )
    
    print(f"\nEntraînement NAFNet terminé !")
    print(f"- Le meilleur modèle a été préservé dans : '{MODEL_SAVE_PATH}'")
    print(f"- L'historique complet a été écrit dans : '{CSV_HISTORY_PATH}'")