"""
train_cnn.py
------------
Entraînement d'un CNN custom (from scratch) avec TensorFlow/Keras
pour la classification d'images en 4 catégories :
  0 = Photo | 1 = Schematics | 2 = Sketch | 3 = Text

Usage :
    python train_cnn.py --csv /chemin/vers/dataset.csv
                        [--epochs 30]
                        [--batch_size 32]
                        [--output_dir ./saved_model]
                        [--val_split 0.2]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, regularizers

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
IMG_SIZE    = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = {0: "Photo", 1: "Schematics", 2: "Sketch", 3: "Text"}

# ──────────────────────────────────────────────
# Arguments CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement CNN - Classification d'images")
    parser.add_argument("--csv",         required=True,          help="Chemin vers le fichier CSV (path,label,split)")
    parser.add_argument("--epochs",      type=int,   default=30, help="Nombre d'époques (défaut : 30)")
    parser.add_argument("--batch_size",  type=int,   default=32, help="Taille du batch (défaut : 32)")
    parser.add_argument("--output_dir",  default="./saved_model", help="Dossier de sauvegarde du modèle")
    parser.add_argument("--val_split",   type=float, default=0.2, help="Proportion de validation sur train (défaut : 0.2)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Chargement & validation du CSV
# ──────────────────────────────────────────────
def load_csv(csv_path: str):
    df = pd.read_csv(csv_path, header=None, names=["path", "label", "split"])
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
    ], name="CNN_Custom")

    return model


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # ── GPU ─────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPU(s) détecté(s) : {[g.name for g in gpus]}")
    else:
        print("[AVERTISSEMENT] Aucun GPU détecté, entraînement sur CPU.")

    # ── CSV ──────────────────────────────────────
    train_df = load_csv(args.csv)

    # Split train / validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_df["path"].values,
        train_df["label"].values,
        test_size=args.val_split,
        stratify=train_df["label"].values,
        random_state=42,
    )
    print(f"[INFO] Train : {len(train_paths)} | Validation : {len(val_paths)}")

    # ── Datasets ─────────────────────────────────
    train_ds = build_dataset(train_paths, train_labels,
                             args.batch_size, augment_data=True, shuffle=True)
    val_ds   = build_dataset(val_paths,   val_labels,
                             args.batch_size, augment_data=False, shuffle=False)

    # ── Modèle ───────────────────────────────────
    model = build_cnn(NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.keras")

    cb_list = [
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
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        # Réduction du learning rate sur plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # Logs TensorBoard (optionnel : tensorboard --logdir ./logs)
        callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, "logs"),
            histogram_freq=1,
        ),
    ]

    # ── Entraînement ─────────────────────────────
    print("\n[INFO] Début de l'entraînement...")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=cb_list,
    )

    # ── Sauvegarde finale ─────────────────────────
    final_path = os.path.join(args.output_dir, "final_model.keras")
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