# doc_field_extraction
Производственная практика 2026 МТУСИ - проект по обучению и анализу языковых моделей для извлечения и анализа текста сканов документов 
Прототип системы извлечения структурированных полей из сканов документов с использованием подходов Document Understanding.
Проект разработан в рамках производственной практики по заданию №24: **«Извлечение полей из сканов документов»**.
## Описание проекта
Цель проекта — разработать прототип, который анализирует сканированный документ, выделяет текстовые блоки, определяет поля формы и устанавливает связи между вопросами и ответами.
Система позволяет:
* работать со сканами документов из набора данных FUNSD;
* анализировать JSON-разметку документов;
* выделять сущности классов `question`, `answer`, `header`, `other`;
* определять связи `question → answer`;
* сохранять результат в структурированном JSON-формате;
* визуализировать найденные поля на изображении документа;
* формировать статистику в JSON и Excel;
* сравнивать архитектуры Document Understanding;
* запускать демонстрационный веб-интерфейс на Streamlit.
## Испльзуемый датасет
В проекте используется датасет **FUNSD** — Form Understanding in Noisy Scanned Documents.
Датасет содержит сканированные формы и JSON-разметку, включающую:
* текстовые блоки;
* координаты bounding box;
* классы сущностей;
* связи между элементами документа.
Используемые классы:

question - вопрос или название поля
answer - ответ или значение поля
header - заголовок или раздел документа
other - прочий текст
## Рассматриваемые архитектуры
В рамках проекта рассмотрены пять архитектур Document Understanding:
| Архитектура | Особенности                                         |
| ----------- | --------------------------------------------------- |
| LayoutLM    | Использует OCR-текст и координаты блоков            |
| LayoutLMv2  | Использует текст, координаты и визуальные признаки  |
| LayoutLMv3  | Унифицированная text-image архитектура              |
| LiLT        | Language-independent Layout Transformer             |
| Donut       | OCR-free подход, работающий напрямую с изображением |

По результатам сравнения основной архитектурой для прототипа выбрана **LayoutLMv3**, так как она показывает хороший баланс качества, скорости обработки и удобства интеграции.
## Структура проекта
```text
doc_field_extraction/
│
├── data/
│   ├── raw/
│   │   └── FUNSD/
│   ├── processed/
│   └── splits/
│
├── models/
│
├── notebooks/
│
├── results/
│   ├── extracted_fields/
│   ├── extracted_visualizations/
│   ├── visualizations/
│   ├── baseline_metrics.json
│   ├── baseline_metrics.xlsx
│   ├── dataset_statistics.json
│   ├── dataset_statistics.xlsx
│   ├── model_comparison.json
│   └── model_comparison.xlsx
│
├── src/
│   ├── check_dataset.py
│   ├── prepare_dataset.py
│   ├── dataset_statistics.py
│   ├── visualize_annotation.py
│   ├── extract_fields_baseline.py
│   ├── visualize_extracted_fields.py
│   ├── evaluate_baseline.py
│   ├── create_model_comparison.py
│   └── interface.py
│
└── README.md
```
## Установка и запуск
### 1. Создание виртуального окружения
```powershell
python -m venv .venv
```
Активация окружения в PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
Если запуск скриптов запрещён:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Установка зависимостей
```powershell
python -m pip install --upgrade pip
pip install numpy pandas matplotlib pillow opencv-python scikit-learn tqdm openpyxl streamlit
```
## Подготовка данных
Датасет FUNSD необходимо разместить в каталоге:
```text
data/raw/FUNSD/
```
После распаковки структура должна быть следующей:
```text
data/raw/FUNSD/
├── training_data/
│   ├── annotations/
│   └── images/
└── testing_data/
    ├── annotations/
    └── images/
```
## Проврка датасета
```powershell
python src\check_dataset.py
```
Скрипт проверяет наличие изображений и JSON-аннотаций в обучающей и тестовой выборках.
## Подготовка train/validation/test
```powershell
python src\prepare_dataset.py
```
После выполнения будут созданы файлы:
```text
data/processed/train.json
data/processed/val.json
data/processed/test.json
data/processed/all.json
```
## Формирование статистики датасета
```powershell
python src\dataset_statistics.py
```
Результаты сохраняются в:
```text
results/dataset_statistics.json
results/dataset_statistics.xlsx
```
## Визуализация исходной разметки
```powershell
python src\visualize_annotation.py
```
Результаты сохраняются в:
```text
results/visualizations/
```
На изображении отображаются:
| Цвет          | Класс               |
| ------------- | ------------------- |
| Синий         | question            |
| Зелёный       | answer              |
| Оранжевый     | header              |
| Серый         | other               |
| Красная линия | связь между блоками |

## Извлечение полей question-answer
```powershell
python src\extract_fields_baseline.py
```
Результт сохраняется в:
```text
results/extracted_fields/
```
Пример выходного JSON:
```json
{
  "document_id": "0000971160",
  "image_path": "data/raw/FUNSD/testing_data/images/0000971160.png",
  "model": "baseline_ground_truth",
  "processing_time_ms": 0.42,
  "fields": [
    {
      "question": "Name",
      "answer": "John Smith",
      "question_bbox": [72, 120, 180, 145],
      "answer_bbox": [210, 120, 330, 145],
      "confidence": 1.0
    }
  ],
  "statistics": {
    "total_entities": 35,
    "total_relations": 12,
    "extracted_fields": 9
  }
}
```
## Визуализация извлечённых полей
```powershell
python src\visualize_extracted_fields.py
```
Результаты сохраняются в:
```text
results/extracted_visualizations/
```
## Оценка baseline-подхода
```powershell
python src\evaluate_baseline.py
```
Результаты сохраняются в:
```text
results/baseline_metrics.json
results/baseline_metrics.xlsx
```
Оцениваются:
* Entity Precision;
* Entity Recall;
* Entity F1;
* Relation Precision;
* Relation Recall;
* Relation F1;
* среднее время обработки документа.
## Сравнение архитектур
```powershell
python src\create_model_comparison.py
```
Результаты сохраняются в:

```text
results/model_comparison.json
results/model_comparison.xlsx
```

Сравниваются архитектуры:

* LayoutLM;
* LayoutLMv2;
* LayoutLMv3;
* LiLT;
* Donut.

## Запуск веб-интерфейса

```powershell
streamlit run src\interface.py
```

После запуска интерфейс будет доступен в браузере:

```text
http://localhost:8501
```
В интерфейсе доступны:
* выбор документа;
* выбор архитектуры модели;
* запуск анализа;
* отображение исходного документа;
* визуализация найденных полей;
* таблица найденных пар question-answer;
* краткая статистика;
* просмотр структурированного JSON;
* скачивание JSON-результата;
* скачивание изображения с визуализацией.
## Основные результаты
В результате выполнения проекта был разработан прототип анализа сканированных документов, который:
* выделяет пары `question → answer`;
* возвращает результат в JSON;
* отображает найденные поля на изображении;
* сохраняет статистику в Excel;
* предоставляет демонстрационный веб-интерфейс;
* содержит сравнение пяти архитектур Document Understanding.

## Используемые технологии
* Python;
* Streamlit;
* pandas;
* OpenCV;
* Pillow;
* scikit-learn;
* openpyxl;
* JSON;
* Excel;
* FUNSD Dataset.

## Возможные направления развития
В дальнейшем проект можно улучшить следующими способами:
* добавить реальное дообучение моделей LayoutLM/LayoutLMv3 на FUNSD;
* подключить OCR-модуль для обработки произвольных пользовательских документов;
* добавить поддержку русскоязычных документов;
* расширить набор данных;
* улучшить алгоритм определения связей question-answer;
* добавить экспорт результатов в PDF;
* реализовать REST API для интеграции с внешними системами.

## Автор
Андрианов Иван Алексеевич
Группа: УБВТ2302, МТУСИ, кафедра «Программная инженерия» 2026.
