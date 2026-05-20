import tensorflow as tf
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
CSV_PATH = "/home/bbapt/cesi/dataset/livrable1/dataset_split.csv"
MODEL_PATH = "/home/bbapt/cesi/liv1/model/fakv8/fakv8_model_final_best.h5"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16

# Correspondance des labels
LABEL_NAMES = {
    0: "Photo",
    1: "Schematics",
    2: "Sketch",
    3: "Text",
    4: "Painting"
}

# --- FONCTION DE PRÉ-TRAITEMENT ---
def load_and_preprocess_image(path):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0
        return image
    except Exception as e:
        return None

# --- CHARGEMENT DU MODÈLE ET DES DONNÉES ---
print(f"Chargement du modèle : {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Lecture du fichier CSV...")
df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    print("Erreur : Aucune donnée avec le split 'test' n'a été trouvée.")
    exit()

print(f"Nombre d'images à tester : {len(test_df)}")

# --- PRÉDICTIONS ---
# On crée un dataset simple (uniquement les images, pas de shuffle pour garder l'ordre)
test_paths = test_df["path"].values
test_labels = test_df["label"].values

def data_generator():
    for path in test_paths:
        img = load_and_preprocess_image(path)
        if img is not None:
            yield img

test_ds = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=tf.TensorSpec(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Exécution des prédictions en cours...")
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# --- ANALYSE DES RÉSULTATS ---
test_df["predicted"] = predicted_classes
test_df["is_correct"] = test_df["predicted"] == test_df["label"]

total_tested = len(test_df)
total_correct = test_df["is_correct"].sum()
total_false = total_tested - total_correct

# --- AFFICHAGE DES RÉSULTATS ---
print("\n" + "="*40)
print("       RAPPORT D'ÉVALUATION DU MODÈLE")
print("="*40)
print(f"Modèle testé : {MODEL_PATH}")
print(f"Image size : {IMAGE_SIZE}, Batch size : {BATCH_SIZE}")

# 1) Total total d'images testé + Total par catégorie
print(f"\n1) STATISTIQUES GLOBALES :")
print(f"   - Total d'images testées : {total_tested}")
for label_id, name in LABEL_NAMES.items():
    count = len(test_df[test_df["label"] == label_id])
    print(f"   - Total {name:10} : {count}")

# 2) Juste vs Faux (Global)
print(f"\n2) PRÉCISION GLOBALE :")
print(f"   - RÉPONSES JUSTES : {total_correct} / {total_tested} ({total_correct/total_tested:.2%})")

print(f"\n4) ERREURS GLOBALES :")
print(f"   - RÉPONSES FAUSSES : {total_false} / {total_tested} ({total_false/total_tested:.2%})")

# 3) Juste vs Faux (Par catégorie)
print(f"\n3) DÉTAILS PAR CATÉGORIE :")
print(f"{'Catégorie':<15} | {'Juste':<12} | {'Faux':<12}")
print("-" * 45)

for label_id, name in LABEL_NAMES.items():
    cat_data = test_df[test_df["label"] == label_id]
    cat_total = len(cat_data)
    
    if cat_total > 0:
        cat_correct = cat_data["is_correct"].sum()
        cat_false = cat_total - cat_correct
        
        juste_str = f"{cat_correct}/{cat_total}"
        faux_str = f"{cat_false}/{cat_total}"
        
        print(f"{name:<15} | {juste_str:<12} | {faux_str:<12}")
    else:
        print(f"{name:<15} | Aucune donnée test")

print("\n" + "="*40)