import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DATA_DIR = Path("data/raw/FUNSD")
OUTPUT_DIR = Path("results/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LABEL_COLORS = {
    "question": "blue",
    "answer": "green",
    "header": "orange",
    "other": "gray",
}


def find_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def draw_annotation(annotation_path: Path, image_path: Path, output_path: Path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    form = data.get("form", [])

    id_to_item = {item["id"]: item for item in form}

    # Рисуем блоки
    for item in form:
        box = item["box"]
        label = item.get("label", "other")
        text = item.get("text", "")

        color = LABEL_COLORS.get(label, "red")

        x1, y1, x2, y2 = box

        draw.rectangle(box, outline=color, width=3)

        caption = f"{label}: {text[:25]}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        tx1, ty1, tx2, ty2 = text_bbox

        draw.rectangle([tx1, ty1, tx2 + 4, ty2 + 4], fill="white")
        draw.text((x1 + 2, y1 + 2), caption, fill=color, font=font)

    # Рисуем связи question-answer
    for item in form:
        source_id = item["id"]
        source_box = item["box"]

        for link in item.get("linking", []):
            if len(link) != 2:
                continue

            a, b = link

            if a != source_id:
                continue

            target_item = id_to_item.get(b)
            if target_item is None:
                continue

            start = get_center(source_box)
            end = get_center(target_item["box"])

            draw.line([start, end], fill="red", width=2)

    image.save(output_path)
    print(f"Сохранено: {output_path}")


def main():
    annotations_dir = DATA_DIR / "training_data" / "annotations"
    images_dir = DATA_DIR / "training_data" / "images"

    annotation_files = sorted(annotations_dir.glob("*.json"))

    if not annotation_files:
        print("Аннотации не найдены")
        return

    annotation_path = annotation_files[0]
    image_path = find_image_by_stem(images_dir, annotation_path.stem)

    if image_path is None:
        print(f"Изображение для {annotation_path.name} не найдено")
        return

    output_path = OUTPUT_DIR / f"{annotation_path.stem}_visualized.png"

    print("Аннотация:", annotation_path)
    print("Изображение:", image_path)

    draw_annotation(annotation_path, image_path, output_path)


if __name__ == "__main__":
    main()