from pathlib import Path
import csv
import numpy as np

# =========================
# CHEMIN DU DATASET
# =========================
_candidates = [
    Path("./data"),
    Path("/tf/notebooks/data"),
]
DATA_DIR = next((p for p in _candidates if p.is_dir()), None)

if DATA_DIR is None:
    raise FileNotFoundError(
        "Dataset introuvable. Placez les images dans un dossier 'data/' "
        "à la racine du projet."
    )

# CSV généré
OUTPUT_CSV = DATA_DIR / "dataset_split.csv"

# Classes — ordre alphabétique pour correspondre à image_dataset_from_directory
CLASS_NAMES = sorted(["Photo", "Schematics", "Sketch", "Text", "Painting"])

# 80% train / 10% val / 10% test
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO = ce qui reste (0.10)

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

    # Split train / val / test
    n = len(files)
    split_train = int(n * TRAIN_RATIO)
    split_val   = int(n * VAL_RATIO)

    train_files = files[:split_train]
    val_files   = files[split_train:split_train + split_val]
    test_files  = files[split_train + split_val:]

    for p in train_files:
        rows.append([str(p), label, "train"])

    for p in val_files:
        rows.append([str(p), label, "val"])

    for p in test_files:
        rows.append([str(p), label, "test"])

# =========================
# ÉCRITURE DU CSV
# =========================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerow(["path", "label", "split"])
    writer.writerows(rows)

n_train = sum(1 for r in rows if r[2] == "train")
n_val   = sum(1 for r in rows if r[2] == "val")
n_test  = sum(1 for r in rows if r[2] == "test")
total   = len(rows)

print(f"CSV créé : {OUTPUT_CSV}")
print(f"Total : {total} images")
print(f"  train : {n_train} ({n_train/total:.0%})")
print(f"  val   : {n_val}   ({n_val/total:.0%})")
print(f"  test  : {n_test}  ({n_test/total:.0%})")
