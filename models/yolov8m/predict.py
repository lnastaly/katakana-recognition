import os
from ultralytics import YOLO

# Wczytanie modelu
model = YOLO('runs/detect/train/weights/best.pt')

# Ścieżka do folderu z obrazami
image_folder = '../../images'

# Sprawdzamy, czy folder istnieje
if not os.path.exists(image_folder):
    print(f"Folder {image_folder} nie istnieje.")
    exit()

# Pobranie listy plików obrazów z folderu
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"Brak obrazów w folderze {image_folder}.")
    exit()

# Przetwarzanie każdego obrazu po kolei
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    print(f"Przetwarzanie obrazu: {image_file}")

    results = model.predict(image_path, save=True, save_txt=True)

    result = results[0]
    result.show()
    print(f"Wyniki dla obrazu {image_file}:\n{result}")

print(f"Wszystkie obrazy zostały przetworzone.")
