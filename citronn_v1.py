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

# Taille des patches d'entraînement (permet de créer des tenseurs fixes homogènes)
IMG_SIZE = 256                 # Sert uniquement à l'entraînement (stratégie par patches aléatoires) - les images originales restent intègres, pas de distorsion ni de resize pour le modèle 'citronn' qui accepte des tailles variables
BATCH_SIZE = 16                # Nombre d'images traitées à la fois
EPOCHS = 300                   # Nombre d'aller-retours d'entraînement

DIR_SAVE = "/home/bbapt/cesi/liv2/model/citronn_v1"                                         # Dossier de sauvegarde de Citronn
MODEL_SAVE_PATH = f"{DIR_SAVE}/model_citronn_v1_2.h5"              # Sauvegardera UNIQUEMENT le meilleur
CSV_HISTORY_PATH = f"{DIR_SAVE}/training_history_citronn_v1_2.csv"      # Fichier d'historique des épochs
# ==============================================================================

def load_dataset():
    """Charge les paires d'images en mémoire en utilisant la méthode des Patches Aléatoires."""
    X_noise = []
    Y_clean = []
    
    print("Chargement et alignement du dataset (Stratégie par Patches)...")
    
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
                
                if img_clean is None or img_noisy is None:
                    continue
                    
                h, w, _ = img_clean.shape
                
                # Extraction d'un patch de 256x256 au hasard (pas de distorsion ni de resize)
                if h >= IMG_SIZE and w >= IMG_SIZE:
                    top = np.random.randint(0, h - IMG_SIZE)
                    left = np.random.randint(0, w - IMG_SIZE)
                    
                    patch_clean = img_clean[top:top+IMG_SIZE, left:left+IMG_SIZE]
                    patch_noisy = img_noisy[top:top+IMG_SIZE, left:left+IMG_SIZE]
                else:
                    # Sécurité si une image source est plus petite que 256x256
                    patch_clean = cv2.resize(img_clean, (IMG_SIZE, IMG_SIZE))
                    patch_noisy = cv2.resize(img_noisy, (IMG_SIZE, IMG_SIZE))
                
                X_noise.append(patch_noisy / 255.0)
                Y_clean.append(patch_clean / 255.0)
                
    return np.array(X_noise), np.array(Y_clean)

def build_citronn(input_shape):
    """Définit l'architecture U-Net 'Citronn' acceptant des tailles variables."""
    input_img = layers.Input(shape=input_shape)
    
    # --- ENCODEUR (Descente) ---
    # Bloc 1
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1) 
    
    # Bloc 2
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv2) 
    
    # --- BOTTLENECK (Le fond du U) ---
    bottleneck = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    
    # --- DÉCODEUR (Remontée avec Skip Connections) ---
    # Bloc 2 inversé
    up2 = layers.UpSampling2D((2, 2))(bottleneck)
    concat2 = layers.Concatenate()([up2, conv2]) 
    conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)
    
    # Bloc 1 inversé
    up1 = layers.UpSampling2D((2, 2))(conv3)
    concat1 = layers.Concatenate()([up1, conv1]) 
    conv4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
    
    # --- COUCHE DE SORTIE ---
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv4)
    
    unet = models.Model(input_img, decoded)
    return unet

def load_keras3_weights(model, weights_path):
    """
    Charge un fichier .weights.h5 sauvegardé avec Keras 3 dans un modèle TF2.
    Keras 3 stocke les poids sous layers/conv2d_N/vars/{0,1}.
    TF2 attribue des noms différents selon le compteur global de couches,
    donc on charge par position plutôt que par nom.
    """
    import h5py
    with h5py.File(weights_path, 'r') as f:
        grp = f['layers']
        # Toutes les couches Conv2D dans l'ordre de création (tri lexicographique sur conv2d_N)
        conv_keys = sorted(
            [k for k in grp.keys() if k.startswith('conv2d')],
            key=lambda k: int(k.split('_')[1]) if '_' in k else 0
        )
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]

        if len(conv_keys) != len(conv_layers):
            raise ValueError(
                f"Incompatibilité : {len(conv_keys)} groupes dans le fichier "
                f"vs {len(conv_layers)} couches Conv2D dans le modèle."
            )

        for layer, key in zip(conv_layers, conv_keys):
            kernel = grp[key]['vars']['0'][:]
            bias   = grp[key]['vars']['1'][:]
            layer.set_weights([kernel, bias])
            print(f"  {layer.name:20s} <- {key}  kernel={kernel.shape}")


if __name__ == "__main__":
    WEIGHTS_PATH = "./model_weights.weights.h5"
    OUTPUT_PATH  = "./model_citron.h5"

    model = build_citronn(input_shape=(None, None, 3))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    print(f"Chargement des poids depuis : {WEIGHTS_PATH}")
    load_keras3_weights(model, WEIGHTS_PATH)

    model.save(OUTPUT_PATH)
    print(f"\nModèle sauvegardé : {OUTPUT_PATH}")