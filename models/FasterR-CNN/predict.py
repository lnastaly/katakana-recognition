import os
import torch
from torchvision.ops import nms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_PATH = "faster_rcnn_model.pth"
CLASS_NAMES = ['a', 'n', 'no', 'nu', 'shi', 'so', 'su', 'ta', 'tsu', 'ya']
IMAGES_DIR = "../../images"

# Ładowanie modelu
model = fasterrcnn_resnet50_fpn(weights=None)
num_classes = len(CLASS_NAMES) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
model.eval()

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# Funkcja, która filtruję predykcje za pomocą nms (non-maximum suppression) i confidence threshold, aby ograniczyć ilość predykcji w jednym miejscu
def apply_nms_with_confidence(predictions, confidence_threshold=0.5, iou_threshold=0.3):
    boxes = torch.tensor(predictions["boxes"])
    scores = torch.tensor(predictions["scores"])
    labels = torch.tensor(predictions["labels"])

    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    keep_indices = nms(boxes, scores, iou_threshold)

    filtered_predictions = {
        "boxes": boxes[keep_indices].numpy(),
        "labels": labels[keep_indices].numpy(),
        "scores": scores[keep_indices].numpy()
    }
    return filtered_predictions

# Funkcja do wyświetlania obrazów z wynikami
def visualize_predictions(image, predictions, threshold=0.3):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score < threshold:
            continue

        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label_text = f"{CLASS_NAMES[label - 1]}: {score:.2f}"
        ax.text(xmin, ymin, label_text, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

def predict_single_image(image_path):
    image = load_image(image_path)
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    predictions = {
        "boxes": outputs[0]["boxes"].cpu().numpy(),
        "labels": outputs[0]["labels"].cpu().numpy(),
        "scores": outputs[0]["scores"].cpu().numpy()
    }

    predictions = apply_nms_with_confidence(predictions, confidence_threshold=0.5, iou_threshold=0.3)
    visualize_predictions(image, predictions)

def predict_images(directory_path):
    if not os.path.exists(directory_path):
        print(f"Folder {directory_path} nie istnieje!")
        return

    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Brak obrazów w folderze: {directory_path}")
        return

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        print(f"\nPrzetwarzanie obrazu: {image_path}")
        predict_single_image(image_path)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Nie znaleziono modelu: {MODEL_PATH}")
    else:
        predict_images(IMAGES_DIR)
