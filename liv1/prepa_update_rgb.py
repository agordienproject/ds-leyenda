import os
from PIL import Image, ImageFile

# Permet de charger même les images tronquées ou légèrement endommagées
ImageFile.LOAD_TRUNCATED_IMAGES = True

def clean_dataset(root_path):
    categories = ['Painting', 'Photo', 'Schematics', 'Sketch', 'Text']
    extensions = ('.jpg', '.jpeg', '.png')
    
    print(f"--- Début du nettoyage du dataset dans : {root_path} ---")
    
    for category in categories:
        folder_path = os.path.join(root_path, category)
        if not os.path.exists(folder_path):
            print(f" [!] Dossier {category} introuvable, on passe...")
            continue
            
        print(f"\nTraitement du dossier : {category}")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
        
        count = 0
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            try:
                # 1. Ouvrir l'image
                with Image.open(file_path) as img:
                    # 2. Convertir en RGB (enlève les profils ICC et gère les PNG transparents)
                    # On convertit pour s'assurer que le format est standard pour le CNN
                    img_clean = img.convert('RGB')
                    
                    # 3. Réenregistrer l'image (écrase le fichier avec une version propre)
                    img_clean.save(file_path, format=None) # format=None garde le format d'origine
                
                count += 1
                if count % 100 == 0:
                    print(f"  > {count} images traitées...")
                    
            except Exception as e:
                print(f"  [X] Impossible de soigner {filename} : {e}")

    print("\n--- Nettoyage terminé ! ---")

# Remplace par le chemin réel (path) vers le dossier contenant tes dossiers Painting, Photo, etc.
dataset_root = '/home/bbapt/cesi/dataset/livrable1' 
clean_dataset(dataset_root)