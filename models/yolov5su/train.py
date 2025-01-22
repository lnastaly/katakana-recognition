import os
import pandas as pd
from ultralytics import YOLO

# Wczytanie modelu YOLOv5
model = YOLO('yolov5su.pt')

# Trening modelu
model.train(data='data.yaml', epochs=20, max_det=1)

# Wczytanie wyników z pliku CSV
results_dir = os.path.join(os.getcwd(), 'runs', 'detect', 'train')
results_csv_path = os.path.join(results_dir, 'results.csv')

results = pd.read_csv(results_csv_path)

# Obliczanie F1 ze wzoru
results['metrics/F1(B)'] = 2 * (results['metrics/precision(B)'] * results['metrics/recall(B)']) / (
    results['metrics/precision(B)'] + results['metrics/recall(B)']
)
results['metrics/F1(B)'] = results['metrics/F1(B)'].fillna(0)

# Metryki są zapisywane w folderze 'results'
output_dir = os.path.abspath(os.path.join(os.getcwd(), '../../results'))
# Folder zostanie utworzony, jeżeli nie istnieje
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'yolov5su_summary.txt')

# Zapisanie metryk do pliku 'yolov5su_summary.txt'
with open(output_file, 'w') as f:
    f.write("Epoch\tPrecision\tRecall\tF1-Score\n")
    for _, row in results.iterrows():
        f.write(f"{int(row['epoch'])}\t{row['metrics/precision(B)']:.4f}\t{row['metrics/recall(B)']:.4f}\t{row['metrics/F1(B)']:.4f}\n")

print(f"Metryki zapisano w: {output_file}")
