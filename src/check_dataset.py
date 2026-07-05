import json
from pathlib import Path

DATA_DIR = Path("data/raw/FUNSD")

train_annotations = DATA_DIR / "training_data" / "annotations"
train_images = DATA_DIR / "training_data" / "images"

test_annotations = DATA_DIR / "testing_data" / "annotations"
test_images = DATA_DIR / "testing_data" / "images"

print("=== TRAIN ===")
print("Папка с аннотациями:", train_annotations.exists())
print("Папка с изображениями:", train_images.exists())

train_annotation_files = list(train_annotations.glob("*.json"))
train_image_files = list(train_images.glob("*"))

print("Количество JSON-разметок:", len(train_annotation_files))
print("Количество изображений:", len(train_image_files))

print("\n=== TEST ===")
print("Папка с аннотациями:", test_annotations.exists())
print("Папка с изображениями:", test_images.exists())

test_annotation_files = list(test_annotations.glob("*.json"))
test_image_files = list(test_images.glob("*"))

print("Количество JSON-разметок:", len(test_annotation_files))
print("Количество изображений:", len(test_image_files))

if train_annotation_files:
    sample_file = train_annotation_files[0]
    print("\nПример JSON-файла:", sample_file.name)

    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Ключи JSON:", list(data.keys()))

    if "form" in data:
        print("Количество элементов form:", len(data["form"]))

        first_item = data["form"][0]
        print("\nПервый элемент form:")
        print(json.dumps(first_item, indent=2, ensure_ascii=False))