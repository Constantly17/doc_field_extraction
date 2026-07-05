import json
from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_document_stats(docs):
    rows = []

    for doc in docs:
        label_counts = {
            "question": 0,
            "answer": 0,
            "header": 0,
            "other": 0,
        }

        for entity in doc["entities"]:
            label = entity["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        rows.append({
            "document_id": doc["document_id"],
            "split": doc["split"],
            "entities_count": len(doc["entities"]),
            "relations_count": len(doc["relations"]),
            "questions": label_counts.get("question", 0),
            "answers": label_counts.get("answer", 0),
            "headers": label_counts.get("header", 0),
            "others": label_counts.get("other", 0),
            "image_path": doc["image_path"],
        })

    return pd.DataFrame(rows)


def collect_split_stats(document_df):
    split_stats = document_df.groupby("split").agg(
        documents=("document_id", "count"),
        entities=("entities_count", "sum"),
        relations=("relations_count", "sum"),
        questions=("questions", "sum"),
        answers=("answers", "sum"),
        headers=("headers", "sum"),
        others=("others", "sum"),
        avg_entities_per_doc=("entities_count", "mean"),
        avg_relations_per_doc=("relations_count", "mean"),
    ).reset_index()

    split_stats["avg_entities_per_doc"] = split_stats["avg_entities_per_doc"].round(2)
    split_stats["avg_relations_per_doc"] = split_stats["avg_relations_per_doc"].round(2)

    return split_stats


def collect_label_stats(docs):
    counts = {}

    for doc in docs:
        for entity in doc["entities"]:
            label = entity["label"]
            counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())

    rows = []
    for label, count in sorted(counts.items()):
        rows.append({
            "label": label,
            "count": count,
            "percent": round(count / total * 100, 2) if total else 0,
        })

    return pd.DataFrame(rows)


def main():
    all_path = PROCESSED_DIR / "all.json"

    if not all_path.exists():
        print("Файл data/processed/all.json не найден.")
        print("Сначала запусти:")
        print("python src\\prepare_dataset.py")
        return

    docs = load_json(all_path)

    document_df = collect_document_stats(docs)
    split_df = collect_split_stats(document_df)
    label_df = collect_label_stats(docs)

    excel_path = RESULTS_DIR / "dataset_statistics.xlsx"
    json_path = RESULTS_DIR / "dataset_statistics.json"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        split_df.to_excel(writer, sheet_name="splits", index=False)
        label_df.to_excel(writer, sheet_name="labels", index=False)
        document_df.to_excel(writer, sheet_name="documents", index=False)

    stats_json = {
        "splits": split_df.to_dict(orient="records"),
        "labels": label_df.to_dict(orient="records"),
        "documents_total": len(document_df),
        "entities_total": int(document_df["entities_count"].sum()),
        "relations_total": int(document_df["relations_count"].sum()),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)

    print("Статистика готова.")
    print(f"Excel: {excel_path}")
    print(f"JSON:  {json_path}")
    print()
    print("Разбиение:")
    print(split_df.to_string(index=False))
    print()
    print("Классы:")
    print(label_df.to_string(index=False))


if __name__ == "__main__":
    main()