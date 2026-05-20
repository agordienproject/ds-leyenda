from pathlib import Path

DATA_DIR = Path("/home/bbapt/cesi/dataset/livrable1")

extensions = set()

for file in DATA_DIR.rglob("*"):

    if file.is_file():

        extensions.add(file.suffix.lower())

print("Extensions trouvées :")

for ext in sorted(extensions):
    print(ext)