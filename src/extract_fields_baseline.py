import json
import time
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/extracted_fields")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def build_entity_map(entities):
    return {entity["id"]: entity for entity in entities}


def extract_question_answer_pairs(doc):
    entities = doc["entities"]
    relations = doc["relations"]

    entity_map = build_entity_map(entities)

    fields = []

    for relation in relations:
        from_id = relation["from_id"]
        to_id = relation["to_id"]

        source = entity_map.get(from_id)
        target = entity_map.get(to_id)

        if source is None or target is None:
            continue

        source_label = source.get("label")
        target_label = target.get("label")

        if source_label == "question" and target_label == "answer":
            question = source
            answer = target
        elif source_label == "answer" and target_label == "question":
            question = target
            answer = source
        else:
            continue

        field = {
            "question_id": question["id"],
            "answer_id": answer["id"],
            "question": normalize_text(question.get("text", "")),
            "answer": normalize_text(answer.get("text", "")),
            "question_bbox": question.get("box", []),
            "answer_bbox": answer.get("box", []),
            "confidence": 1.0
        }

        fields.append(field)

    return fields


def build_document_result(doc, model_name="baseline_ground_truth"):
    start_time = time.perf_counter()

    fields = extract_question_answer_pairs(doc)

    end_time = time.perf_counter()
    processing_time_ms = round((end_time - start_time) * 1000, 3)

    label_counts = {
        "question": 0,
        "answer": 0,
        "header": 0,
        "other": 0,
    }

    for entity in doc["entities"]:
        label = entity.get("label", "other")
        label_counts[label] = label_counts.get(label, 0) + 1

    result = {
        "document_id": doc["document_id"],
        "image_path": doc["image_path"],
        "model": model_name,
        "processing_time_ms": processing_time_ms,
        "fields": fields,
        "statistics": {
            "total_entities": len(doc["entities"]),
            "total_relations": len(doc["relations"]),
            "extracted_fields": len(fields),
            "labels": label_counts
        }
    }

    return result


def main():
    test_path = PROCESSED_DIR / "test.json"

    if not test_path.exists():
        print("Файл data/processed/test.json не найден.")
        print("Сначала запусти:")
        print("python src\\prepare_dataset.py")
        return

    test_docs = load_json(test_path)

    if not test_docs:
        print("В test.json нет документов")
        return

    doc = test_docs[0]

    result = build_document_result(doc)

    output_path = OUTPUT_DIR / f"{doc['document_id']}_fields.json"
    save_json(output_path, result)

    print("Документ:", result["document_id"])
    print("Модель:", result["model"])
    print("Время обработки, мс:", result["processing_time_ms"])
    print("Всего сущностей:", result["statistics"]["total_entities"])
    print("Всего связей:", result["statistics"]["total_relations"])
    print("Извлечено полей:", result["statistics"]["extracted_fields"])
    print()
    print("Найденные поля:")

    for i, field in enumerate(result["fields"], start=1):
        print(f"{i}. {field['question']} -> {field['answer']}")

    print()
    print(f"JSON сохранён: {output_path}")


if __name__ == "__main__":
    main()