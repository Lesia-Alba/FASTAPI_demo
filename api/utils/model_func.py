from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
BERT_DIR = (BASE_DIR / "weights" / "bert_ru_base_model").resolve()
YOLO_PATH = (BASE_DIR / "weights" / "yolo11" / "best_brain.pt").resolve()


def load_text_model():
    """
    Загружает tokenizer и BERT-модель один раз.
    Возвращает словарь с нужными объектами.
    """
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BERT_DIR),
        local_files_only=True
    )
    model.eval()

    return {
        "tokenizer": tokenizer,
        "model": model,
    }


def load_image_model():
    """
    Загружает YOLO-модель один раз.
    """
    model = YOLO(str(YOLO_PATH))
    return model


def predict_text(text: str, model_bundle) -> dict:
    """
    Принимает текст и уже загруженную BERT-модель.
    Возвращает предсказанный класс и вероятность.
    """
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        prob = probs[0][pred_id].item()

    id2label = model.config.id2label

    custom_labels = {
        "LABEL_0": "negative",
        "LABEL_1": "positive",
    }

    raw_label = id2label.get(pred_id, str(pred_id))
    label = custom_labels.get(raw_label, raw_label)

    return {
        "label": label,
        "prob": round(prob, 4),
    }


def predict_image(image_bytes: bytes, model) -> dict:
    """
    Принимает байты изображения и уже загруженную YOLO-модель.
    Возвращает список найденных объектов.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    medical_labels = {
        "negative": "no tumor detected",
        "positive": "tumor detected",
    }

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()

        class_name = results[0].names[class_id]
        class_name = medical_labels.get(class_name, class_name)

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x, 2) for x in xyxy],
            }
        )

    return {"detections": detections}