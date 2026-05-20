"""
train_cnn.py
------------
Entraînement d'un CNN custom (from scratch) avec TensorFlow/Keras
pour la classification d'images en 5 catégories :
  0 = Photo | 1 = Schematics | 2 = Sketch | 3 = Text | 4 = Painting

Usage :
    python train_cnn.py
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, regularizers

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  — modifiez uniquement cette section
# ══════════════════════════════════════════════════════════════════

# Chemin vers le fichier CSV (path,label,split)
CSV_PATH    = "/home/bbapt/cesi/dataset/livrable1/dataset_split.csv"

# Dossier où seront sauvegardés le modèle et les logs TensorBoard
OUTPUT_DIR  = "/home/bbapt/cesi/liv1/model"

# Taille des images en entrée (hauteur, largeur)
IMG_SIZE    = (224, 224)

# Nombre d'époques maximum (l'EarlyStopping peut arrêter avant)
EPOCHS      = 30

# Nombre d'images traitées par batch
BATCH_SIZE  = 16

# Proportion du train utilisée pour la validation (ex: 0.2 = 20 %)
VAL_SPLIT   = 0.2

# Learning rate initial de l'optimiseur Adam
LEARNING_RATE = 1e-3

# ── Callbacks ────────────────────────────────────────────────────
# EarlyStopping : arrêt si val_accuracy ne progresse plus après N époques
EARLY_STOPPING_PATIENCE = 7

# ReduceLROnPlateau : divise le LR si val_loss stagne après N époques
REDUCE_LR_PATIENCE      = 3
REDUCE_LR_FACTOR        = 0.5
REDUCE_LR_MIN           = 1e-6

# ── Classes ───────────────────────────────────────────────────────
NUM_CLASSES = 5
CLASS_NAMES = {0: "Photo", 1: "Schematics", 2: "Sketch", 3: "Text", 4: "Painting"}

# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# Chargement & validation du CSV
# ──────────────────────────────────────────────
def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)  # le CSV a une ligne d'en-tête (path, label, split)
    df.columns = ["path", "label", "split"]  # renommage explicite pour garantir les noms
    df["label"] = df["label"].astype(int)
 
    # Vérifications
    invalid_labels = df[~df["label"].isin(CLASS_NAMES.keys())]
    if not invalid_labels.empty:
        print(f"[AVERTISSEMENT] {len(invalid_labels)} ligne(s) avec label inconnu ignorée(s).")
        df = df[df["label"].isin(CLASS_NAMES.keys())]
 
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"[INFO] Lignes 'train' trouvées : {len(train_df)}")
 
    # Distribution des classes
    for label, name in CLASS_NAMES.items():
        count = (train_df["label"] == label).sum()
        print(f"       {name:>12} (label {label}) : {count} images")
 
    return train_df

# ──────────────────────────────────────────────
# Pipeline tf.data
# ──────────────────────────────────────────────
def load_and_preprocess(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # normalisation [0,1]
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_dataset(paths, labels, batch_size, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (list(paths), list(labels))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ──────────────────────────────────────────────
# Architecture CNN custom
# ──────────────────────────────────────────────
def build_cnn(num_classes: int) -> models.Sequential:
    """
    Architecture :
        3 blocs Conv → BatchNorm → ReLU → MaxPool → Dropout
        puis GlobalAveragePooling → Dense(512) → Dense(num_classes)
    """
    model = models.Sequential([
        # ── Bloc 1 ──────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same", input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Bloc 2 ──────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Bloc 3 ──────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Bloc 4 ──────────────────────────────────
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # ── Tête de classification ───────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="CNN_Pik")

    return model


# ──────────────────────────────────────────────
# Mémoire GPU
# ──────────────────────────────────────────────
def print_gpu_memory(label: str = ""):
    """Affiche la mémoire GPU utilisée / pic via tf (nécessite CUDA)."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return
        for i in range(len(gpus)):
            info = tf.config.experimental.get_memory_info(f"GPU:{i}")
            used_mb = info["current"] / 1024 ** 2
            peak_mb = info["peak"]    / 1024 ** 2
            print(f"  [GPU:{i} MEM] {label:30s} | Utilisée : {used_mb:7.1f} Mo  |  Pic : {peak_mb:7.1f} Mo")
    except Exception as e:
        print(f"  [GPU MEM] Impossible de lire la mémoire : {e}")


class GpuMemoryCallback(tf.keras.callbacks.Callback):
    """Affiche la consommation mémoire GPU à la fin de chaque époque."""
    def on_epoch_end(self, epoch, logs=None):
        print_gpu_memory(f"Fin époque {epoch + 1:03d}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # ── GPU ─────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPU(s) détecté(s) : {[g.name for g in gpus]}")
        print_gpu_memory("Avant chargement des données")
    else:
        # Arrêt du script
        print("[AVERTISSEMENT] Aucun GPU détecté, entraînement sur CPU.")
        return

    # ── CSV ──────────────────────────────────────
    train_df = load_csv(CSV_PATH)

    # Split train / validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_df["path"].values,
        train_df["label"].values,
        test_size=VAL_SPLIT,
        stratify=train_df["label"].values,
        random_state=42,
    )
    print(f"[INFO] Train : {len(train_paths)} | Validation : {len(val_paths)}")

    # ── Datasets ─────────────────────────────────
    train_ds = build_dataset(train_paths, train_labels,
                             BATCH_SIZE, augment_data=True, shuffle=True)
    val_ds   = build_dataset(val_paths,   val_labels,
                             BATCH_SIZE, augment_data=False, shuffle=False)

    # ── Modèle ───────────────────────────────────
    model = build_cnn(NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print_gpu_memory("Après chargement du modèle")

    # ── Callbacks ────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")

    cb_list = [
        # Affichage mémoire GPU à chaque époque
        GpuMemoryCallback(),
        # Sauvegarde automatique du meilleur modèle (val_accuracy)
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Arrêt anticipé si la validation ne s'améliore plus
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Réduction du learning rate sur plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        )#,
        # Logs TensorBoard (optionnel : tensorboard --logdir ./saved_model/logs)
        #callbacks.TensorBoard(
        #    log_dir=os.path.join(OUTPUT_DIR, "logs"),
        #    histogram_freq=1,
        #),
    ]

    # ── Entraînement ─────────────────────────────
    print("\n[INFO] Début de l'entraînement...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=cb_list,
    )

    # ── Sauvegarde finale ─────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "pik_v3_model.keras")
    model.save(final_path)
    print(f"\n[INFO] Modèle final sauvegardé : {final_path}")
    print(f"[INFO] Meilleur modèle sauvegardé : {checkpoint_path}")

    # ── Résumé des métriques ──────────────────────
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch   = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"\n[RÉSULTATS]")
    print(f"  Meilleure val_accuracy : {best_val_acc:.4f} (époque {best_epoch})")
    print(f"  Époques réalisées      : {len(history.history['loss'])}")


if __name__ == "__main__":
    main()