import os
import cv2
import numpy as np
import tensorflow as tf

# ==============================================================================
# VARIABLES IMPORTANTES (À CONFIGURER)
# ==============================================================================
MODEL_PATH = "/home/bbapt/cesi/liv2/model/blur_2/best_model_blur_2.h5"  # Chemin de ton modèle entraîné
TEST_DIR = "/home/bbapt/cesi/dataset/livrable2/test_img"                # Dossier contenant tes images de test
OUTPUT_RESULTS_DIR = "test_results"                                     # Dossier où sauvegarder les visuels "Avant/Après"

IMG_SIZE = 256                                    # Doit être identique à l'entraînement
# ==============================================================================

def calculate_metrics(img_true, img_pred):
    """Calcule la MSE et le PSNR entre l'image de référence et l'image testée."""
    mse = np.mean((img_true - img_pred) ** 2)
    if mse == 0:
        return 0, float('inf')
    
    # Utilisation de la fonction native de OpenCV pour le PSNR
    psnr = cv2.PSNR(img_true, img_pred)
    return mse, psnr

def run_testing():
    # 1. Chargement du modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Le modèle n'existe pas à l'emplacement '{MODEL_PATH}'")
        return
    
    print(f"Chargement du modèle : {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
    
    # Dictionnaires pour stocker les stats par niveau de bruit
    stats_by_level = {}
    
    # Extensions valides
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files = os.listdir(TEST_DIR)
    
    # Filtre pour ne prendre que les images bruitées
    noisy_files = [f for f in all_files if "_noise_" in f and f.lower().endswith(valid_extensions)]
    
    if not noisy_files:
        print(f"Aucune image bruitée (contenant '_noise_') trouvée dans '{TEST_DIR}'.")
        return

    print(f"Détection de {len(noisy_files)} images bruitées. Début du débruitage...")

    for noisy_name in noisy_files:
        # Extraction du nom de base et du niveau de bruit (ex: paysage_noise_50.jpg)
        base_name, rest = noisy_name.split("_noise_")
        noise_level = rest.split(".")[0]  # Récupère "50"
        ext = "." + rest.split(".")[-1]   # Récupère ".jpg"
        clean_name = base_name + ext       # Reconstruit "paysage.jpg"
        
        noisy_path = os.path.join(TEST_DIR, noisy_name)
        clean_path = os.path.join(TEST_DIR, clean_name)
        
        # Vérification de l'existence de l'image de référence
        if not os.path.exists(clean_path):
            print(f"⚠️ Image de référence manquante pour : {noisy_name} (Attendu : {clean_name})")
            continue
            
        # 2. Chargement et redimensionnement
        img_clean = cv2.imread(clean_path)
        img_noisy = cv2.imread(noisy_path)
        
        img_clean_resized = cv2.resize(img_clean, (IMG_SIZE, IMG_SIZE))
        img_noisy_resized = cv2.resize(img_noisy, (IMG_SIZE, IMG_SIZE))
        
        # Prépare l'image pour le modèle (Normalisation + Ajout de la dimension Batch)
        input_tensor = np.expand_dims(img_noisy_resized / 255.0, axis=0)
        
        # 3. Prédiction par l'IA
        denoised_tensor = model.predict(input_tensor, verbose=0)[0]
        
        # Denormalisation pour revenir en pixels standards (0-255 uint8)
        img_denoised = (denoised_tensor * 255).astype(np.uint8)
        
        # 4. Calcul des scores
        mse_avant, psnr_avant = calculate_metrics(img_clean_resized, img_noisy_resized)
        mse_apres, psnr_apres = calculate_metrics(img_clean_resized, img_denoised)
        
        # Stockage des stats pour le bilan final
        if noise_level not in stats_by_level:
            stats_by_level[noise_level] = {'psnr_in': [], 'psnr_out': [], 'mse_in': [], 'mse_out': []}
            
        stats_by_level[noise_level]['psnr_in'].append(psnr_avant)
        stats_by_level[noise_level]['psnr_out'].append(psnr_apres)
        stats_by_level[noise_level]['mse_in'].append(mse_avant)
        stats_by_level[noise_level]['mse_out'].append(mse_apres)
        
        # 5. Sauvegarde d'un visuel comparatif côte à côte (Propre | Bruité | Débruité)
        comparison_visual = np.hstack([img_clean_resized, img_noisy_resized, img_denoised])
        
        # On écrit les scores PSNR directement sur l'image pour l'analyse visuelle
        cv2.putText(comparison_visual, f"Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(comparison_visual, f"Bruit: {psnr_avant:.1f}dB", (IMG_SIZE + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(comparison_visual, f"IA: {psnr_apres:.1f}dB", (IMG_SIZE * 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        output_name = f"result_{base_name}_{noise_level}{ext}"
        cv2.imwrite(os.path.join(OUTPUT_RESULTS_DIR, output_name), comparison_visual)

    # ==============================================================================
    # AFFICHAGE DU BILAN STATISTIQUE FINALE
    # ==============================================================================
    print("\n" + "="*60)
    print("         BILAN DES PERFORMANCES DU MODÈLE AI")
    print("="*60)
    print(f"{'Niveau Bruit':<15} | {'PSNR Initial':<14} | {'PSNR Final (IA)':<16} | {'Gain (dB)':<10}")
    print("-"*60)
    
    # Tri des niveaux de bruit par valeur numérique
    for level in sorted(stats_by_level.keys(), key=int):
        avg_psnr_in = np.mean(stats_by_level[level]['psnr_in'])
        avg_psnr_out = np.mean(stats_by_level[level]['psnr_out'])
        gain = avg_psnr_out - avg_psnr_in
        
        print(f"{level:<15} | {avg_psnr_in:<14.2f} | {avg_psnr_out:<16.2f} | {f'{gain:.2f}':<10}")
    print("="*60)
    print(f"Les images comparatives ont été sauvegardées dans : '{OUTPUT_RESULTS_DIR}/'")

if __name__ == "__main__":
    run_testing()