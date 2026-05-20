import os
import cv2
import numpy as np

# ==============================================================================
# VARIABLES IMPORTANTES (À CONFIGURER)
# ==============================================================================
#SOURCE_DIR = "/home/bbapt/cesi/dataset/livrable2/source"    # Dossier contenant tes images propres
SOURCE_DIR = "/home/bbapt/cesi/dataset/livrable2/test_img"
OUTPUT_DIR = "/home/bbapt/cesi/dataset/livrable2/test_img_tempo"     # Dossier où seront sauvegardées les images bruitées

# Niveaux de bruit à appliquer (0.08 = 8% de bruit, 0.2 = 20%, 0.5 = 50%)
# Le pourcentage définit l'écart-type (sigma) du bruit par rapport à l'intensité max (255)
NOISE_LEVELS = [0.08, 0.12, 0.2, 0.5, 0.7]

# Mode de génération :
# True  -> Génère 3 images bruitées pour CHAQUE image propre (Dossier final plus gros)
# False -> Choisit un niveau de bruit aléatoire parmi la liste pour chaque image (Gain de place)
MULTIPLY_IMAGES = True
# ==============================================================================

def add_gaussian_noise(image, percentage):
    """Ajoute du bruit gaussien à une image en fonction d'un pourcentage d'intensité."""
    row, col, ch = image.shape
    mean = 0
    # sigma correspond à l'écart-type du bruit
    sigma = percentage * 255
    
    # Génération du bruit gaussien
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    
    # Application du bruit et clip pour rester entre 0 et 255
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    return noisy

def process_dataset():
    # Vérification et création du dossier de sortie
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Dossier '{OUTPUT_DIR}' créé.")

    # Extensions d'images acceptées
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Aucune image trouvée dans {SOURCE_DIR}. Vérifie le chemin ou les extensions.")
        return

    print(f"Début du traitement de {len(image_files)} images...")

    for img_name in image_files:
        img_path = os.path.join(SOURCE_DIR, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Impossible de lire l'image : {img_name}")
            continue

        name, ext = os.path.splitext(img_name)

        if MULTIPLY_IMAGES:
            # On applique chaque niveau de bruit à l'image
            for level in NOISE_LEVELS:
                noisy_img = add_gaussian_noise(image, level)
                # On renomme l'image pour savoir quel niveau de bruit est appliqué
                output_name = f"{name}_noise_{int(level*100)}{ext}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, output_name), noisy_img)
        else:
            # On choisit un niveau au hasard pour cette image
            random_level = np.random.choice(NOISE_LEVELS)
            noisy_img = add_gaussian_noise(image, random_level)
            output_name = f"{name}_noise_{int(random_level*100)}{ext}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, output_name), noisy_img)

    print(f"Traitement terminé ! Les images sont dans le dossier '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    # Petit check de sécurité pour éviter que le script plante direct
    if not os.path.exists(SOURCE_DIR):
        print(f"Erreur : Le dossier source '{SOURCE_DIR}' n'existe pas. Crée-le ou modifie la variable en haut du script.")
    else:
        process_dataset()