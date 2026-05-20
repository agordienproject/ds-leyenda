import tensorflow as tf
import pandas as pd
import os

# --- CONFIGURATION DES PARAMÈTRES ---
CSV_PATH = "/home/bbapt/cesi/dataset/livrable1/dataset_split.csv" 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 5 
MODEL_SAVE_NAME = "fakv1_model.h5"

# --- FONCTION DE SURVEILLANCE GPU ---
def print_gpu_utilization():
    """Affiche la mémoire utilisée par le GPU (NVIDIA)."""
    print("\n--- État de la mémoire GPU ---")
    # Cette commande fonctionne pour les GPU NVIDIA (ex: RTX 2060)
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
        print(f"Erreur sur l'image {path}: {e}")
        return None

print("Chargement du CSV...")
# Lecture simplifiée car le CSV possède déjà un header 'path,label,split'
df = pd.read_csv(CSV_PATH)

# Filtrage pour ne garder que les données d'entraînement
train_df = df[df["split"] == "train"]

print(f"Nombre d'images pour l'entraînement : {len(train_df)}")

# Création du Dataset TensorFlow
list_ds = tf.data.Dataset.from_tensor_slices((train_df["path"].values, train_df["label"].values))

# Transformation en dataset d'images avec exécution parallèle
train_ds = list_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Optimisation du flux de données
train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- ARCHITECTURE DU MODÈLE CNN ---
# Modèle adapté pour la classification en 5 catégories (0 à 4)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACK POUR LE GPU ---
class GPUUsageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print_gpu_utilization()

# --- ENTRAÎNEMENT ---
print("Début de l'entraînement...")
print_gpu_utilization()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[GPUUsageCallback()]
)

# --- SAUVEGARDE ---
model.save(MODEL_SAVE_NAME)
print(f"\nEntraînement terminé. Modèle enregistré sous : {MODEL_SAVE_NAME}")