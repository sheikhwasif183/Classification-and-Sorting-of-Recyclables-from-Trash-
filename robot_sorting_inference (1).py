from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

bin_map = {
    'cardboard': 'recyclable',
    'glass': 'recyclable',
    'metal': 'recyclable',
    'paper': 'recyclable',
    'plastic': 'recyclable',
    'trash': 'general_waste'
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    return model


def load_model(weights_path):
    device = get_device()

    model = build_model()
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model, device


def map_class_to_bin(class_name):
    return bin_map[class_name]


def predict_pil_image(model, image, device):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, pred_index = torch.max(probabilities, dim=1)

    pred_index = pred_index.item()
    confidence = confidence.item()

    pred_class = class_names[pred_index]
    pred_bin = map_class_to_bin(pred_class)

    return pred_class, pred_bin, confidence


def predict_image_path(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    return predict_pil_image(model, image, device)


def predict_frame(model, frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return predict_pil_image(model, image, device)


def webcam_demo(model, device, camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started.")
    print("Press SPACE to classify the current frame.")
    print("Press Q to quit.")

    last_text = "Waiting for prediction..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, last_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Robot Sorting Demo", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            pred_class, pred_bin, confidence = predict_frame(model, frame, device)
            last_text = f"Class: {pred_class} | Bin: {pred_bin} | Conf: {confidence:.4f}"
            print(last_text)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="efficientnet_b0_best.pth")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model, device = load_model(weights_path)

    print("Model loaded successfully.")
    print(f"Device: {device}")

    if args.image is not None:
        pred_class, pred_bin, confidence = predict_image_path(model, args.image, device)
        print(f"Predicted class: {pred_class}")
        print(f"Final bin: {pred_bin}")
        print(f"Confidence: {confidence:.4f}")

    elif args.webcam:
        webcam_demo(model, device, camera_index=args.camera_index)

    else:
        print("No input selected.")
        print("Use --image path/to/image.jpg or --webcam")


if __name__ == "__main__":
    main()