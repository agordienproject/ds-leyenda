import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" # opti chat gpt

import tensorflow as tf
import pandas as pd

# Chat gpt opti
tf.config.optimizer.set_jit(False)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.backend.clear_session()

# --- CONFIGURATION DES PARAMÈTRES ---
CSV_PATH = "/home/bbapt/cesi/dataset/livrable1/dataset_split.csv" 
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 50                 # Augmenté à 50 car l'EarlyStopping gère la coupure automatique
LEARNING_RATE = 0.001
NUM_CLASSES = 5 
VALIDATION_SPLIT = 0.2      # 0.2 = 20% des données 'train' serviront à la validation

# Configuration des dossiers et noms de sauvegarde
SAVE_DIR = "model/fakv7"
os.makedirs(SAVE_DIR, exist_ok=True)  # Crée le dossier 'model/fakv7' s'il n'existe pas

# --
shuffle_eco_start = True    # Permet de réduire la charge GPU en limitant le shuffle à un buffer plus petit
shuffle_eco_buffer = 500    # Taille du buffer de shuffle pour économiser la mémoire GPU (si shuffle_eco_start est True)
shuffle_eco_batch = 2       # Nombre de batches à précharger pour économiser la mémoire GPU (si shuffle_eco_start est True)

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
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        return image, label
    except Exception as e:
        return None

print("Chargement du CSV...")
df = pd.read_csv(CSV_PATH)

full_train_df = df[df["split"] == "train"].copy()
full_train_df = full_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

split_idx = int(len(full_train_df) * (1 - VALIDATION_SPLIT))

train_df = full_train_df.iloc[:split_idx]
val_df = full_train_df.iloc[split_idx:]

print(f"Total images 'train' dans le CSV : {len(full_train_df)}")
print(f"Images utilisées pour l'entraînement pur : {len(train_df)}")
print(f"Images utilisées pour la validation interne : {len(val_df)}")

def prepare_dataset(dataframe, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((dataframe["path"].values, dataframe["label"].values))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

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
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    
    # Augmentation
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    
    # Bloc 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    # Bloc 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    # Bloc 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    
    # Bloc 4
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    # Tête de classification
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACKS OPTIMISÉS ---

class GPUUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print_gpu_utilization()

# 1. Arrêt précoce si la perte de validation ne baisse plus pendant 6 époques
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True, # Recharge automatiquement les meilleurs poids à la fin
    verbose=1
)

# 2. Réduction du Learning Rate (divisé par 5) si stagnation pendant 3 époques
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# 3. Sauvegarde UNIQUEMENT du meilleur modèle (au format .h5 ou .keras)
checkpoint_path = os.path.join(SAVE_DIR, "fakv7_best_model.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,       # Évite d'écrire sur le disque à chaque époque si pas d'amélioration
    mode='min',
    verbose=1
)

# 4. Enregistrement de l'historique en direct dans le CSV (remplace ton code Pandas de fin)
csv_save_path = os.path.join(SAVE_DIR, "training_history.csv")
csv_logger = tf.keras.callbacks.CSVLogger(
    csv_save_path,
    separator=',',
    append=False
)

# Regroupement de tous les callbacks
callbacks_list = [GPUUsageCallback(), early_stopping, reduce_lr, checkpoint_callback, csv_logger]

# --- ENTRAÎNEMENT ---
print("Début de l'entraînement...")
print_gpu_utilization()

history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=callbacks_list  
)

# --- SAUVEGARDE FINALE DE SÉCURITÉ ---
# Comme EarlyStopping a déjà restauré les meilleurs poids dans "model", 
# le sauvegarder ici revient à enregistrer une copie conforme du meilleur modèle.
final_save_path = os.path.join(SAVE_DIR, "fakv7_model_final_best.h5")
model.save(final_save_path)

print(f"\nEntraînement terminé de manière optimale.")
print(f"Meilleur modèle sauvegardé sous : {checkpoint_path}")
print(f"Historique CSV mis à jour sous : {csv_save_path}")