import tensorflow as tf
import pandas as pd
import os

# --- CONFIGURATION DES PARAMÈTRES ---
CSV_PATH = "/home/bbapt/cesi/dataset/livrable1/dataset_split.csv" 
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 5 
VALIDATION_SPLIT = 0.2      # 0.2 = 20% des données 'train' serviront à la validation

# Configuration des dossiers et noms de sauvegarde
SAVE_DIR = "model/fakv5"
os.makedirs(SAVE_DIR, exist_ok=True)  # Crée le dossier 'model/fakv5' s'il n'existe pas

# --
shuffle_eco_start = True    # Permet de réduire la charge GPU en limitant le shuffle à un buffer plus petit
shuffle_eco_buffer = 500    # Taille du buffer de shuffle pour économiser la mémoire GPU (si shuffle_eco_start est True)
shuffle_eco_batch = 2       # Nombre de batches à précharger pour économiser la mémoire GPU (si shuffle_eco_start est True)

# --- FONCTION DE SURVEILLANCE GPU ---
def print_gpu_utilization():
    """Affiche la mémoire utilisée par le GPU (NVIDIA)."""
    print("\n--- État de la mémoire GPU ---")
    # Commande pour GPU NVIDIA (RTX 2060)
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
        # En cas d'erreur sur une image, on peut l'afficher ici
        return None

print("Chargement du CSV...")
df = pd.read_csv(CSV_PATH)

# On isole toutes les données destinées à l'entraînement d'après le CSV
full_train_df = df[df["split"] == "train"].copy()

# Mélange aléatoire des données avant le split
full_train_df = full_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calcul de l'index de séparation pour la validation (Correction ici)
split_idx = int(len(full_train_df) * (1 - VALIDATION_SPLIT))

train_df = full_train_df.iloc[:split_idx]
val_df = full_train_df.iloc[split_idx:]

print(f"Total images 'train' dans le CSV : {len(full_train_df)}")
print(f"Images utilisées pour l'entraînement pur : {len(train_df)}")
print(f"Images utilisées pour la validation interne : {len(val_df)}")

# Fonction pour créer un Dataset TensorFlow optimisé
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

# Création des deux jeux de données
train_ds = prepare_dataset(train_df, shuffle=True)
val_ds = prepare_dataset(val_df, shuffle=False)

# Supprimer full_train_df pour libérer de la mémoire
del full_train_df

# --- ARCHITECTURE DU MODÈLE CNN ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    
    # Bloc 1 : 32 filtres
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Bloc 2 : 64 filtres
    tf.keras.layers.Conv2D(64, (3, 3),strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Bloc 3 : 128 filtres
    tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Bloc 4 : 256 filtres (Pour plus de détails)
    tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Flatten
    tf.keras.layers.Flatten(),
    
    # Couche Dense plus efficace
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACKS ---
class GPUUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print_gpu_utilization()

# Callback de sauvegarde par époque
checkpoint_path = os.path.join(SAVE_DIR, "fakv5_model_e{epoch}.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,  
    verbose=1
)

# --- ENTRAÎNEMENT ---
print("Début de l'entraînement...")
print_gpu_utilization()

history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=EPOCHS,
    callbacks=[GPUUsageCallback(), checkpoint_callback]  
)

# --- SAUVEGARDE FINALE ---
final_save_path = os.path.join(SAVE_DIR, "fakv5_model_eFinal.h5")
model.save(final_save_path)
print(f"\nEntraînement terminé. Modèle final enregistré sous : {final_save_path}")