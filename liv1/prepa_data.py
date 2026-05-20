from pathlib import Path
import csv
import numpy as np

# =========================
# CHEMIN DU DATASET
# =========================
DATA_DIR = Path("/home/bbapt/cesi/dataset/livrable1")

# CSV généré
OUTPUT_CSV = DATA_DIR / "dataset_split.csv"

# Classes
CLASS_NAMES = ["Photo", "Schematics", "Sketch", "Text", "Painting"]

# 20% pour le test
TEST_RATIO = 0.20

# Extensions acceptées
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Générateur aléatoire
SEED = 42
rng = np.random.default_rng(SEED)

rows = []

# =========================
# PARCOURS DES DOSSIERS
# =========================
for label, class_name in enumerate(CLASS_NAMES):

    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {class_dir}")

    files = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if len(files) == 0:
        raise ValueError(f"Aucune image trouvée dans : {class_dir}")

    files = np.array(files)

    # Mélange aléatoire
    rng.shuffle(files)

    # Split train / test
    split_idx = int(len(files) * (1 - TEST_RATIO))

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Ajout TRAIN
    for p in train_files:
        rows.append([str(p), label, "train"])

    # Ajout TEST
    for p in test_files:
        rows.append([str(p), label, "test"])

# =========================
# ÉCRITURE DU CSV
# =========================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(["path", "label", "split"])

    writer.writerows(rows)

print(f"CSV créé : {OUTPUT_CSV}")
print(f"Nombre total d'images : {len(rows)}")