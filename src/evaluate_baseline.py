import json
import time
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_relation(relation):
    return tuple(sorted((relation["from_id"], relation["to_id"])))


def evaluate_ground_truth_baseline(docs):
    """
    Baseline использует истинную разметку FUNSD.
    Поэтому предсказания полностью совпадают с ground truth.
    """

    total_entities_true = 0
    total_entities_pred = 0
    total_entities_correct = 0

    total_relations_true = 0
    total_relations_pred = 0
    total_relations_correct = 0

    processing_times = []

    document_rows = []

    for doc in docs:
        start_time = time.perf_counter()

        true_entities = doc["entities"]
        pred_entities = doc["entities"]

        true_relations = doc["relations"]
        pred_relations = doc["relations"]

        true_entity_set = set(
            (entity["id"], entity["label"], tuple(entity["box"]))
            for entity in true_entities
        )

        pred_entity_set = set(
            (entity["id"], entity["label"], tuple(entity["box"]))
            for entity in pred_entities
        )

        true_relation_set = set(normalize_relation(r) for r in true_relations)
        pred_relation_set = set(normalize_relation(r) for r in pred_relations)

        correct_entities = len(true_entity_set & pred_entity_set)
        correct_relations = len(true_relation_set & pred_relation_set)

        end_time = time.perf_counter()
        processing_time_ms = round((end_time - start_time) * 1000, 4)

        processing_times.append(processing_time_ms)

        total_entities_true += len(true_entity_set)
        total_entities_pred += len(pred_entity_set)
        total_entities_correct += correct_entities

        total_relations_true += len(true_relation_set)
        total_relations_pred += len(pred_relation_set)
        total_relations_correct += correct_relations

        document_rows.append({
            "document_id": doc["document_id"],
            "split": doc["split"],
            "true_entities": len(true_entity_set),
            "pred_entities": len(pred_entity_set),
            "correct_entities": correct_entities,
            "true_relations": len(true_relation_set),
            "pred_relations": len(pred_relation_set),
            "correct_relations": correct_relations,
            "processing_time_ms": processing_time_ms,
        })

    entity_precision = total_entities_correct / total_entities_pred if total_entities_pred else 0
    entity_recall = total_entities_correct / total_entities_true if total_entities_true else 0
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if entity_precision + entity_recall > 0
        else 0
    )

    relation_precision = total_relations_correct / total_relations_pred if total_relations_pred else 0
    relation_recall = total_relations_correct / total_relations_true if total_relations_true else 0
    relation_f1 = (
        2 * relation_precision * relation_recall / (relation_precision + relation_recall)
        if relation_precision + relation_recall > 0
        else 0
    )

    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

    summary = {
        "model": "baseline_ground_truth",
        "documents": len(docs),

        "entity_precision": round(entity_precision, 4),
        "entity_recall": round(entity_recall, 4),
        "entity_f1": round(entity_f1, 4),

        "relation_precision": round(relation_precision, 4),
        "relation_recall": round(relation_recall, 4),
        "relation_f1": round(relation_f1, 4),

        "avg_processing_time_ms": round(avg_time, 4),

        "total_entities_true": total_entities_true,
        "total_entities_pred": total_entities_pred,
        "total_entities_correct": total_entities_correct,

        "total_relations_true": total_relations_true,
        "total_relations_pred": total_relations_pred,
        "total_relations_correct": total_relations_correct,
    }

    return summary, document_rows


def main():
    test_path = PROCESSED_DIR / "test.json"

    if not test_path.exists():
        print("Файл data/processed/test.json не найден.")
        print("Сначала запусти:")
        print("python src\\prepare_dataset.py")
        return

    test_docs = load_json(test_path)

    summary, document_rows = evaluate_ground_truth_baseline(test_docs)

    json_path = RESULTS_DIR / "baseline_metrics.json"
    excel_path = RESULTS_DIR / "baseline_metrics.xlsx"

    save_json(json_path, summary)

    summary_df = pd.DataFrame([summary])
    document_df = pd.DataFrame(document_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        document_df.to_excel(writer, sheet_name="documents", index=False)

    print("Оценка baseline завершена.")
    print()
    print("Метрики:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print(f"JSON сохранён:  {json_path}")
    print(f"Excel сохранён: {excel_path}")


if __name__ == "__main__":
    main()