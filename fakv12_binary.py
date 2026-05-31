import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # opti chat gpt

import json
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Chat gpt opti
tf.config.optimizer.set_jit(False)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.backend.clear_session()
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- CONFIGURATION DES PARAMÈTRES ---
CSV_PATH = "data/fakv12_labels.csv" 
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 50                 # Augmenté à 50 car l'EarlyStopping gère la coupure automatique
LEARNING_RATE = 0.001
NUM_CLASSES = 1             # MODIFIÉ : 1 seule sortie pour de la classification binaire
VALIDATION_SPLIT = 0.2      # 0.2 = 20% des données 'train' serviront à la validation

# Configuration des dossiers et noms de sauvegarde
SAVE_DIR = "model/fakv12_binary"
NAME_SUFFIX = "fakv12_binary"
os.makedirs(SAVE_DIR, exist_ok=True)  

# --
shuffle_eco_start = True    
shuffle_eco_buffer = 500    
shuffle_eco_batch = 2       

# --- FONCTION DE SURVEILLANCE GPU ---
def print_gpu_utilization():
    """Affiche la mémoire utilisée par le GPU (NVIDIA)."""
    print("\n--- État de la mémoire GPU ---")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader")
    print("------------------------------\n")

# --- PRÉPARATION DES DONNÉES ---
def load_and_preprocess_image(path, label):
    try:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        return image, label
    except Exception as e:
        return None

print("Chargement du CSV...")
df = pd.read_csv(CSV_PATH)

# Transformation binaire des labels
# Si le label d'origine est 0 (Photo) -> devient 1 (Classe positive)
# Si le label d'origine est 1, 2, 3 ou 4 -> devient 0 (Classe négative / "Pas une photo")
df["label"] = (df["label"] == 0).astype(int)

full_train_df = df[df["split"] == "train"].copy()

# MODIFICATION 1 : Séparation Stratifiée pour garder le bon ratio 1:3 dans train et val
train_df, val_df = train_test_split(
    full_train_df, 
    test_size=VALIDATION_SPLIT, 
    stratify=full_train_df["label"], 
    random_state=42
)

print(f"Total images 'train' dans le CSV : {len(full_train_df)}")
print(f"Images utilisées pour l'entraînement pur : {len(train_df)}")
print(f"Images utilisées pour la validation interne : {len(val_df)}")

# MODIFICATION 2 : Calcul automatique des poids des classes pour compenser le déséquilibre
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Poids des classes appliqués pour l'entraînement : {class_weight_dict}")

def prepare_dataset(dataframe, shuffle=False):
    # Les labels ici sont maintenant des 0 et des 1
    ds = tf.data.Dataset.from_tensor_slices((dataframe["path"].values, dataframe["label"].values))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if shuffle_eco_start:   
        if shuffle:     
            ds = ds.shuffle(buffer_size=shuffle_eco_buffer)
        ds = ds.batch(BATCH_SIZE).prefetch(buffer_size = shuffle_eco_batch)
    else:
        if shuffle: 
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
        
    return ds

train_ds = prepare_dataset(train_df, shuffle=True)
val_ds = prepare_dataset(val_df, shuffle=False)

del full_train_df

# --- ARCHITECTURE DU MODÈLE CNN ---
def residual_block(x, filters):
    """
    Crée un bloc résiduel type ResNet.
    L'entrée 'x' contourne les convolutions et est additionnée à la sortie.
    """
    shortcut = x
    
    # 1ère convolution standard (remplace le SeparableConv2D)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # 2ème convolution standard
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Si le nombre de filtres change entre l'entrée et la sortie, 
    # on doit ajuster la dimension du shortcut avec une Conv 1x1
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    # L'astuce ResNet : on additionne l'entrée originale et le résultat des convolutions
    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# --- Construction du modèle avec l'API Fonctionnelle ---
inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Augmentation de données
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)

# Bloc 1 : Entrée (Standard)
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)

# Bloc 2 : Résiduel 64 filtres
x = residual_block(x, 64)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)

# Bloc 3 : Résiduel 128 filtres
x = residual_block(x, 128)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)

# Bloc 4 : Résiduel 256 filtres
x = residual_block(x, 256)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)

# Tête de classification équilibrée
x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(128, use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)

# On applique le Dropout UNE SEULE FOIS, fortement, juste avant la sortie
x = tf.keras.layers.Dropout(0.5)(x)

# Sortie binaire — dtype float32 obligatoire avec mixed precision
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid', dtype='float32')(x)

# Instanciation du modèle
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# MODIFICATION 3 : Ajout de métriques robustes pour le déséquilibre
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Afficher un résumé de la nouvelle architecture dans le terminal
model.summary()

# --- CALLBACKS OPTIMISÉS ---

class GPUUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print_gpu_utilization()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5, 
    patience=6, 
    min_lr=1e-6,
    verbose=1
)

checkpoint_path = os.path.join(SAVE_DIR, NAME_SUFFIX + "_best_model.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,      
    mode='min',
    verbose=1
)

callbacks_list = [GPUUsageCallback(), early_stopping, reduce_lr, checkpoint_callback]

# --- ENTRAÎNEMENT ---
print("Début de l'entraînement...")
print_gpu_utilization()

history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=callbacks_list,
    class_weight=class_weight_dict # MODIFICATION 4 : Application des poids ici
)

json_save_path = os.path.join(SAVE_DIR, "training_history.json")
hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(json_save_path, "w") as f:
    json.dump(hist_data, f, indent=1)

print(f"\nEntraînement binaire terminé avec succès.")
print(f"Meilleur modèle sauvegardé sous : {checkpoint_path}")
print(f"Historique JSON sauvegardé sous : {json_save_path}")