import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path("results/extracted_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS_DIR = Path("results/extracted_fields")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def draw_label(draw, box, text, color, font):
    x1, y1, x2, y2 = box

    draw.rectangle(box, outline=color, width=3)

    caption = text[:40]
    text_bbox = draw.textbbox((x1, max(0, y1 - 18)), caption, font=font)
    tx1, ty1, tx2, ty2 = text_bbox

    draw.rectangle([tx1, ty1, tx2 + 4, ty2 + 4], fill="white")
    draw.text((x1 + 2, max(0, y1 - 18)), caption, fill=color, font=font)


def visualize_fields(fields_json_path: Path):
    result = load_json(fields_json_path)

    image_path = Path(result["image_path"])

    if not image_path.exists():
        print(f"Изображение не найдено: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    fields = result.get("fields", [])

    for index, field in enumerate(fields, start=1):
        question_box = field["question_bbox"]
        answer_box = field["answer_bbox"]

        question_text = f"Q{index}: {field['question']}"
        answer_text = f"A{index}: {field['answer']}"

        draw_label(draw, question_box, question_text, "blue", font)
        draw_label(draw, answer_box, answer_text, "green", font)

        q_center = get_center(question_box)
        a_center = get_center(answer_box)

        draw.line([q_center, a_center], fill="red", width=2)

        middle_x = (q_center[0] + a_center[0]) // 2
        middle_y = (q_center[1] + a_center[1]) // 2

        draw.ellipse(
            [middle_x - 10, middle_y - 10, middle_x + 10, middle_y + 10],
            fill="white",
            outline="red",
            width=2
        )
        draw.text((middle_x - 5, middle_y - 7), str(index), fill="red", font=font)

    output_path = OUTPUT_DIR / f"{result['document_id']}_fields_visualized.png"
    image.save(output_path)

    print("Готово.")
    print(f"Документ: {result['document_id']}")
    print(f"Найдено полей: {len(fields)}")
    print(f"Сохранено: {output_path}")


def main():
    json_files = sorted(FIELDS_DIR.glob("*_fields.json"))

    if not json_files:
        print("Файлы с извлечёнными полями не найдены.")
        print("Сначала запусти:")
        print("python src\\extract_fields_baseline.py")
        return

    fields_json_path = json_files[0]

    print(f"Используется файл: {fields_json_path}")
    visualize_fields(fields_json_path)


if __name__ == "__main__":
    main()