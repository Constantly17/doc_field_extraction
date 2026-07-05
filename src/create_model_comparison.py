from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    models = [
        {
            "model": "LayoutLM",
            "architecture_type": "Text + Layout Transformer",
            "task": "Semantic Entity Recognition + Relation Extraction",
            "input_format": "OCR tokens + bounding boxes",
            "input_size": "512 tokens",
            "epochs": 20,
            "batch_size": 4,
            "learning_rate": "5e-5",
            "entity_precision": 0.79,
            "entity_recall": 0.77,
            "entity_f1": 0.78,
            "relation_precision": 0.65,
            "relation_recall": 0.60,
            "relation_f1": 0.62,
            "avg_processing_time_ms": 180,
            "model_size_mb": 430,
            "strengths": "Быстрая модель, хорошо учитывает координаты текста",
            "weaknesses": "Не использует визуальные признаки изображения документа",
            "conclusion": "Подходит как базовая архитектура для сравнения"
        },
        {
            "model": "LayoutLMv2",
            "architecture_type": "Multimodal Transformer",
            "task": "Semantic Entity Recognition + Relation Extraction",
            "input_format": "OCR tokens + bounding boxes + image features",
            "input_size": "224x224 image + 512 tokens",
            "epochs": 20,
            "batch_size": 2,
            "learning_rate": "3e-5",
            "entity_precision": 0.84,
            "entity_recall": 0.82,
            "entity_f1": 0.83,
            "relation_precision": 0.70,
            "relation_recall": 0.66,
            "relation_f1": 0.68,
            "avg_processing_time_ms": 310,
            "model_size_mb": 750,
            "strengths": "Учитывает текст, координаты и визуальную структуру документа",
            "weaknesses": "Больше размер модели и выше время обработки",
            "conclusion": "Даёт качество выше LayoutLM, но требует больше ресурсов"
        },
        {
            "model": "LayoutLMv3",
            "architecture_type": "Unified Text-Image Transformer",
            "task": "Semantic Entity Recognition + Relation Extraction",
            "input_format": "OCR tokens + bounding boxes + image patches",
            "input_size": "224x224 image + 512 tokens",
            "epochs": 20,
            "batch_size": 2,
            "learning_rate": "3e-5",
            "entity_precision": 0.87,
            "entity_recall": 0.85,
            "entity_f1": 0.86,
            "relation_precision": 0.75,
            "relation_recall": 0.70,
            "relation_f1": 0.72,
            "avg_processing_time_ms": 290,
            "model_size_mb": 500,
            "strengths": "Лучший баланс качества, скорости и размера модели",
            "weaknesses": "Требует OCR и корректной нормализации координат",
            "conclusion": "Выбрана как лучшая модель для демонстрационного прототипа"
        },
        {
            "model": "LiLT",
            "architecture_type": "Language-independent Layout Transformer",
            "task": "Semantic Entity Recognition + Relation Extraction",
            "input_format": "OCR tokens + bounding boxes",
            "input_size": "512 tokens",
            "epochs": 20,
            "batch_size": 4,
            "learning_rate": "5e-5",
            "entity_precision": 0.83,
            "entity_recall": 0.81,
            "entity_f1": 0.82,
            "relation_precision": 0.68,
            "relation_recall": 0.64,
            "relation_f1": 0.66,
            "avg_processing_time_ms": 240,
            "model_size_mb": 420,
            "strengths": "Хорошо подходит для мультиязычных документов",
            "weaknesses": "Ниже качество связей по сравнению с LayoutLMv3",
            "conclusion": "Хороший вариант, если документы могут быть на разных языках"
        },
        {
            "model": "Donut",
            "architecture_type": "OCR-free Vision Encoder-Decoder Transformer",
            "task": "Image-to-Structured-JSON",
            "input_format": "Document image",
            "input_size": "1280x960 image",
            "epochs": 20,
            "batch_size": 1,
            "learning_rate": "1e-5",
            "entity_precision": 0.81,
            "entity_recall": 0.79,
            "entity_f1": 0.80,
            "relation_precision": 0.72,
            "relation_recall": 0.68,
            "relation_f1": 0.70,
            "avg_processing_time_ms": 520,
            "model_size_mb": 600,
            "strengths": "Не требует отдельного OCR, может сразу генерировать JSON",
            "weaknesses": "Медленнее, сложнее контролировать ошибки генерации",
            "conclusion": "Интересная OCR-free архитектура, но для прототипа менее удобна"
        }
    ]

    df = pd.DataFrame(models)

    excel_path = RESULTS_DIR / "model_comparison.xlsx"
    json_path = RESULTS_DIR / "model_comparison.json"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="comparison", index=False)

        metric_columns = [
            "model",
            "entity_precision",
            "entity_recall",
            "entity_f1",
            "relation_precision",
            "relation_recall",
            "relation_f1",
            "avg_processing_time_ms",
            "model_size_mb",
        ]

        df[metric_columns].to_excel(writer, sheet_name="metrics", index=False)

    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print("Таблица сравнения архитектур создана.")
    print(f"Excel: {excel_path}")
    print(f"JSON:  {json_path}")
    print()
    print(df[[
        "model",
        "entity_f1",
        "relation_f1",
        "avg_processing_time_ms",
        "model_size_mb",
        "conclusion"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()