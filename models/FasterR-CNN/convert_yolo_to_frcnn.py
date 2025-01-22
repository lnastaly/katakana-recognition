import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# Ścieżki datasetu w formacie YOLO
yolo_train_dir = '../../datasets/dataset_yolo/images/train'
yolo_val_dir = '../../datasets/dataset_yolo/images/val'
yolo_labels_train_dir = '../../datasets/dataset_yolo/labels/train'
yolo_labels_val_dir = '../../datasets/dataset_yolo/labels/val'

# Ścieżki docelowe dla nowego datasetu
output_train_dir = '/../..datasets/dataset_frcnn/train/images'
output_val_dir = '../../datasets/dataset_frcnn/val/images'
output_train_annotations_dir = '../../datasets/dataset_frcnn/train/annotations'
output_val_annotations_dir = '../../datasets/dataset_frcnn/val/annotations'

os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_train_annotations_dir, exist_ok=True)
os.makedirs(output_val_annotations_dir, exist_ok=True)

class_names = ['a', 'n', 'no', 'nu', 'shi', 'so', 'su', 'ta', 'tsu', 'ya']
class_map = {name: idx + 1 for idx, name in enumerate(class_names)}

# Funkcja konwertuje dataset z formatu YOLO (etykiety z bounding boxami w formacie .txt) na format Pascal VOC (etykiety w formacie XML)
def convert_yolo_to_frcnn(yolo_image_dir, yolo_label_dir, output_image_dir, output_annotation_dir):
    for label_file in os.listdir(yolo_label_dir):
        if label_file.endswith('.txt'):
            image_name_jpg = label_file.replace('.txt', '.jpg')
            image_name_png = label_file.replace('.txt', '.png')

            image_path = None
            if os.path.exists(os.path.join(yolo_image_dir, image_name_jpg)):
                image_path = os.path.join(yolo_image_dir, image_name_jpg)
            elif os.path.exists(os.path.join(yolo_image_dir, image_name_png)):
                image_path = os.path.join(yolo_image_dir, image_name_png)

            if image_path is None:
                print(f"Warning: Image {image_name_jpg} or {image_name_png} not found. Skipping...")
                continue

            image = Image.open(image_path)
            width, height = image.size

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "images"
            ET.SubElement(root, "filename").text = os.path.basename(image_path)
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "Unknown"

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(width)
            ET.SubElement(size, "height").text = str(height)
            ET.SubElement(size, "depth").text = "3"

            with open(os.path.join(yolo_label_dir, label_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    # id klas w YOLO zaczyna od 0, a w Faster R-CNN od 1, dlatego dodajemy tutaj 1
                    class_idx = int(parts[0]) + 1
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    obj_width = float(parts[3]) * width
                    obj_height = float(parts[4]) * height

                    xmin = int(x_center - obj_width / 2)
                    ymin = int(y_center - obj_height / 2)
                    xmax = int(x_center + obj_width / 2)
                    ymax = int(y_center + obj_height / 2)

                    obj = ET.SubElement(root, "object")
                    ET.SubElement(obj, "name").text = class_names[class_idx - 1]
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(xmin)
                    ET.SubElement(bndbox, "ymin").text = str(ymin)
                    ET.SubElement(bndbox, "xmax").text = str(xmax)
                    ET.SubElement(bndbox, "ymax").text = str(ymax)

            tree = ET.ElementTree(root)
            xml_filename = os.path.join(output_annotation_dir, label_file.replace('.txt', '.xml'))
            tree.write(xml_filename)

            shutil.copy(image_path, os.path.join(output_image_dir, os.path.basename(image_path)))


convert_yolo_to_frcnn(yolo_train_dir, yolo_labels_train_dir, output_train_dir, output_train_annotations_dir)
convert_yolo_to_frcnn(yolo_val_dir, yolo_labels_val_dir, output_val_dir, output_val_annotations_dir)
