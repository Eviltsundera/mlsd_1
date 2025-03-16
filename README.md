# Классификация новостных статей по категориям

Проект по многоклассовой классификации новостных статей с использованием трансформеров.

## Описание проекта

Этот проект решает задачу автоматической классификации новостных статей по категориям (например, Бизнес, Технологии, Спорт, Развлечения, Политика). Задача представляет собой многоклассовую классификацию с похожими классами в доменной области текстовых данных.

## Структура проекта 
```
project/
├── data/                  # Директория с данными
│   ├── raw/               # Исходные данные
│   └── processed/         # Обработанные данные
├── models/                # Сохраненные модели
├── notebooks/             # Jupyter ноутбуки
├── src/                   # Исходный код
│   ├── data/              # Скрипты для обработки данных
│   ├── features/          # Скрипты для создания признаков
│   ├── models/            # Скрипты для обучения моделей
│   └── visualization/     # Скрипты для визуализации
├── environment.yml        # Conda окружение
└── README.md              # Документация проекта
```

## Настройка окружения

Проект использует conda для управления зависимостями. Для настройки окружения выполните следующие шаги:

### Предварительные требования

- [Anaconda](https://www.anaconda.com/download/) или [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA-совместимая видеокарта (опционально, для ускорения обучения)

### Создание окружения

1. Клонируйте репозиторий:
```bash
git clone <url-репозитория>
cd <название-репозитория>
```

2. Создайте conda окружение из файла environment.yml:
```bash
conda env create -f environment.yml
```

3. Активируйте окружение:
```bash
conda activate news_classification
```

### Проверка установки

Для проверки корректности установки запустите:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Использование

[Здесь будут инструкции по использованию проекта]

## Структура данных

[Здесь будет описание структуры данных]

## Модели

Проект использует трансформеры для классификации текста. Основные модели:
- BERT
- RoBERTa
- DistilBERT
- XLM-RoBERTa

## Результаты

[Здесь будут результаты экспериментов]

## Лицензия

[Информация о лицензии] 