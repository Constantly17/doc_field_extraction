import json
import random
from pathlib import Path


RANDOM_SEED = 42

DATA_DIR = Path("data/raw/FUNSD")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image_by_stem(images_dir: Path, stem: str) -> str | None:
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate).replace("\\", "/")
    return None


def load_funsd_split(split_name: str, source_folder: str):
    annotations_dir = DATA_DIR / source_folder / "annotations"
    images_dir = DATA_DIR / source_folder / "images"

    documents = []

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        document_id = annotation_path.stem
        image_path = find_image_by_stem(images_dir, document_id)

        if image_path is None:
            print(f"Изображение не найдено для {annotation_path.name}")
            continue

        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities = []
        relations = []

        for item in data.get("form", []):
            entity = {
                "id": item["id"],
                "text": item.get("text", ""),
                "label": item.get("label", "other"),
                "box": item.get("box", []),
                "words": item.get("words", [])
            }

            entities.append(entity)

            for link in item.get("linking", []):
                if len(link) == 2:
                    relations.append({
                        "from_id": link[0],
                        "to_id": link[1]
                    })

        document = {
            "document_id": document_id,
            "split": split_name,
            "image_path": image_path,
            "annotation_path": str(annotation_path).replace("\\", "/"),
            "entities": entities,
            "relations": relations,
            "statistics": {
                "entities_count": len(entities),
                "relations_count": len(relations),
                "labels": {}
            }
        }

        label_counts = {}
        for entity in entities:
            label = entity["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        document["statistics"]["labels"] = label_counts

        documents.append(document)

    return documents


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    random.seed(RANDOM_SEED)

    train_full = load_funsd_split("train_full", "training_data")
    test = load_funsd_split("test", "testing_data")

    random.shuffle(train_full)

    val_size = int(len(train_full) * 0.2)

    val = train_full[:val_size]
    train = train_full[val_size:]

    for doc in train:
        doc["split"] = "train"

    for doc in val:
        doc["split"] = "val"

    save_json(OUTPUT_DIR / "train.json", train)
    save_json(OUTPUT_DIR / "val.json", val)
    save_json(OUTPUT_DIR / "test.json", test)

    all_docs = train + val + test
    save_json(OUTPUT_DIR / "all.json", all_docs)

    print("Готово.")
    print(f"Train: {len(train)} документов")
    print(f"Val:   {len(val)} документов")
    print(f"Test:  {len(test)} документов")
    print(f"All:   {len(all_docs)} документов")

    total_entities = sum(len(doc["entities"]) for doc in all_docs)
    total_relations = sum(len(doc["relations"]) for doc in all_docs)

    print()
    print(f"Всего сущностей: {total_entities}")
    print(f"Всего связей:    {total_relations}")

    global_label_counts = {}

    for doc in all_docs:
        for entity in doc["entities"]:
            label = entity["label"]
            global_label_counts[label] = global_label_counts.get(label, 0) + 1

    print()
    print("Распределение классов:")
    for label, count in sorted(global_label_counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()