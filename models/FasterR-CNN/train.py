import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET
import torch.optim as optim
import numpy as np

# Klasa tworząca dataset z obrazów ze znakami katakany
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        annotation_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".xml").replace(".png", ".xml"))
        boxes, labels = self.parse_xml(annotation_path)

        target = {"boxes": boxes, "labels": labels}
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def parse_xml(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            label = class_names.index(name) + 1
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels

# Funkcja do obliczania metryk precision, recall, F1
def calculate_metrics(preds, targets, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for p, t in zip(preds, targets):
        confusion_matrix[t, p] += 1

    precision = []
    recall = []
    f1 = []

    for cls in range(1, num_classes):
        tp = confusion_matrix[cls, cls]
        fp = confusion_matrix[:, cls].sum() - tp
        fn = confusion_matrix[cls, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cls = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_cls)

    return np.mean(precision), np.mean(recall), np.mean(f1)

# Określanie ścieżek dla zbioru treningowego
base_dir = os.path.dirname(os.path.abspath(__file__))

train_img_dir = os.path.join(base_dir, "../../datasets/dataset_frcnn/train/images")
train_annotation_dir = os.path.join(base_dir, "../../datasets/dataset_frcnn/train/annotations")
val_img_dir = os.path.join(base_dir, "../../datasets/dataset_frcnn/val/images")
val_annotation_dir = os.path.join(base_dir, "../../datasets/dataset_frcnn/val/annotations")

class_names = ['a', 'n', 'no', 'nu', 'shi', 'so', 'su', 'ta', 'tsu', 'ya']

transform = transforms.Compose([transforms.ToTensor()])

# Parametry treningu
train_dataset = CustomDataset(train_img_dir, train_annotation_dir, transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

val_dataset = CustomDataset(val_img_dir, val_annotation_dir, transform)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=len(class_names) + 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Trening modelu
num_epochs = 20
output_dir = os.path.abspath(os.path.join(os.getcwd(), '../../results'))
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'faster_rcnn_summary.txt')

with open(output_file, "w") as f:
    f.write("Epoch\tPrecision\tRecall\tF1-Score\n")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (imgs, targets) in enumerate(train_dataloader):
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        losses.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_dataloader)} Loss: {losses.item():.4f}")

    print(f"[Epoch {epoch+1}/{num_epochs}] Total Loss: {total_loss:.4f}")

    # Ewaluacja po zakończeniu epoki
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in val_dataloader:
            imgs = [img.to(device) for img in imgs]

            outputs = model(imgs)
            for output, target in zip(outputs, targets):
                pred_labels = output['labels'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()

                all_preds.extend(pred_labels)
                all_targets.extend(true_labels)

    precision, recall, f1 = calculate_metrics(all_preds, all_targets, num_classes=len(class_names) + 1)
    print(f"[Epoch {epoch + 1}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Zapisujemy metryki do pliku "faster_rcnn_summary.txt" w folderze results po zakończeniu epoki
    with open(output_file, "a") as f:
        f.write(f"{epoch + 1}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\n")

print(f"Metryki zapisano w: {output_file}")

torch.save(model.state_dict(), "faster_rcnn_model.pth")
print("Model saved as faster_rcnn_model.pth")
