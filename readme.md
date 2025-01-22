# Rozpoznawanie znaków katakany

## Opis projektu
Projekt wykorzystuje modele sztucznej inteligencji do rozpoznawania japońskich znaków katakany. Celem jest stworzenie systemu zdolnego do precyzyjnego klasyfikowania znaków, które często są mylone ze względu na ich podobny wygląd.

Modele zostały wytrenowane, aby rozpoznawać znaki:

| Romanizacja | Katakana |
|-------------|----------|
| a           | ア        |
| n           | ン        |
| no          | ノ        |
| nu          | ヌ        |
| shi         | シ        |
| so          | ソ        |
| su          | ス        |
| ta          | タ        |
| tsu         | ツ        |
| ya          | ヤ        |


## Struktura projektu
```text
Projekt
├── datasets
│   ├── dataset_frcnn
│   ├── dataset_mnv2
│   └── dataset_yolo
├── images
├── models
│   ├── FasterR-CNN
│   ├── MobileNetV2
│   ├── yolov5
│   └── yolov8m
├── readme.md
├── requirements.txt
└── results
    ├── faster_rcnn_summary.txt
    ├── yolov5su_summary.txt
    └── yolov8m_summary.txt
```

W folderze ***datasets*** znajdują się zbiory danych w formatach przeznaczonych dla wykorzystanych w projekcie modeli 

W folderze ***models*** znajdują się podfoldery z użytymi modelami:
- yolov8m
- yolov5su
- FasterR-CNN
- MobileNetV2

W każdym z nich znajdują się skrypty:
- *train.py* – do trenowania modelu
- *predict.py* – do uzyskania predykcji na podstawie przekazanych obrazów

Model MobileNetV2 wykonuje predykcje pojedynczego obrazu, którego ścieżka jest okreslona w zmiennej image_path w pliku *predict.py*. Pozostałe modele wykonują predykcje wszystkich obrazów znajdujących się w folderze ***images***.

W trakcie treningu metryki precision, recall oraz F1-score są zapisywane w folderze ***results*** jako plik tekstowy *"nazwa_modelu_summary.txt"*

## Instrukcja instalacji

1. **Sklonuj repozytorium**:
   ```bash
   git clone https://github.com/twoj-repo/katakana-recognition.git
   cd katakana-recognition
   ```
2. **Zainstaluj wymagane zależności**

   Lista wymaganych zależności znajduje się w pliku *requirements.txt*. Aby je zainstalować można użyć polecenia:
    ```bash
    pip install -r requirements.txt
    ```

## Instrukcja użycia
1. **Wrzuć obrazy ze znakami katakany**
2. **Przejdź do folderu wybranego modelu:**
   ```bash
   cd models/nazwa_modelu
   ```
3. **Użyj pliku predict.py w folderze wybranego modelu, aby wykonać predykcje na obrazach z folderu images:**
   ```bash
   python predict.py
   ```
   *Uwaga: plik predict.py modelu MobileNetV2 obsługuje wykonywanie predykcji tylko na pojedynczym obrazie. Aby zmienić obraz, na którym zostanie wykonana predykcja zmień wartość zmiennej image_path, która określa ścieżkę obrazu*


4. **Predykcje na wybranych obrazach zostaną wyświetlone w osobnych oknach.**

   
