#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для объединения датасетов с гармонизацией меток:
- Загрузка очищенных датасетов
- Приведение меток к единому формату (6 классов)
- Объединение датасетов
- Разделение на обучающую, валидационную и тестовую выборки
- Сохранение объединенного датасета
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from collections import Counter
import re
from sklearn.model_selection import train_test_split

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути к директориям
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CLEAN_DATA_DIR = PROCESSED_DATA_DIR / "clean"
MERGED_DATA_DIR = PROCESSED_DATA_DIR / "merged"

# Создаем директорию для объединенных датасетов
MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Определение соответствия меток между датасетами
# 6 общих категорий
LABEL_MAPPING = {
    "ag_news": {
        0: "world",       # World -> World/Politics
        1: "sports",      # Sports -> Sports
        2: "business",    # Business -> Business
        3: "technology",  # Sci/Tech -> Technology
    },
    "bbc_news": {
        "business": "business",          # Business -> Business
        "entertainment": "entertainment", # Entertainment -> Entertainment
        "politics": "world",             # Politics -> World/Politics
        "sport": "sports",               # Sport -> Sports
        "tech": "technology",            # Tech -> Technology
    },
    "20newsgroups": {
        # Группировка 20 классов в 6 общих категорий
        "alt.atheism": "other",
        "comp.graphics": "technology",
        "comp.os.ms-windows.misc": "technology",
        "comp.sys.ibm.pc.hardware": "technology",
        "comp.sys.mac.hardware": "technology",
        "comp.windows.x": "technology",
        "misc.forsale": "business",
        "rec.autos": "other",
        "rec.motorcycles": "other",
        "rec.sport.baseball": "sports",
        "rec.sport.hockey": "sports",
        "sci.crypt": "technology",
        "sci.electronics": "technology",
        "sci.med": "other",
        "sci.space": "technology",
        "soc.religion.christian": "other",
        "talk.politics.guns": "world",
        "talk.politics.mideast": "world",
        "talk.politics.misc": "world",
        "talk.religion.misc": "other",
    }
}


def load_clean_datasets():
    """Загрузка очищенных датасетов."""
    datasets = {}
    
    for name in ["ag_news", "bbc_news", "20newsgroups"]:
        dataset_path = CLEAN_DATA_DIR / name
        if dataset_path.exists():
            logger.info(f"Загрузка очищенного датасета {name}...")
            try:
                datasets[name] = load_from_disk(str(dataset_path))
                logger.info(f"Датасет {name} успешно загружен")
            except Exception as e:
                logger.error(f"Ошибка при загрузке датасета {name}: {e}")
    
    return datasets


def get_harmonized_labels(data, source, label_col):
    """Получение гармонизированных меток для датасета."""
    if source == "20newsgroups":
        # Для 20newsgroups метки хранятся как числа, но есть колонка label_name
        label_names = data["label_name"].tolist()
        return [LABEL_MAPPING[source][label_name] for label_name in label_names]
    else:
        # Для других датасетов применяем маппинг напрямую
        return [LABEL_MAPPING[source][label] for label in data[label_col]]


def process_split(dataset, split, source, label_col):
    """Обработка одного сплита датасета."""
    if split not in dataset:
        return None
    
    # Получение оригинальных данных
    data = dataset[split].to_pandas()
    
    # Применение маппинга меток
    harmonized_labels = get_harmonized_labels(data, source, label_col)
    
    # Добавление новой колонки с гармонизированными метками
    data["harmonized_label"] = harmonized_labels
    
    # Добавление колонки с источником
    data["source"] = source
    
    # Преобразование обратно в Dataset
    return Dataset.from_pandas(data)


def harmonize_labels(datasets):
    """Гармонизация меток в датасетах."""
    logger.info(f"Гармонизация меток...")
    
    harmonized_datasets = {}
    
    for name, dataset in datasets.items():
        logger.info(f"Обработка датасета {name}...")
        
        # Определение колонки с метками
        label_col = "label"
        if name == "bbc_news":
            label_col = "category"
        
        # Обработка train и test сплитов
        harmonized_train = process_split(dataset, "train", name, label_col)
        harmonized_test = process_split(dataset, "test", name, label_col)
        
        # Создание нового DatasetDict
        harmonized_datasets[name] = DatasetDict({
            "train": harmonized_train,
            "test": harmonized_test
        })
        
        # Вывод статистики по гармонизированным меткам
        if harmonized_train is not None:
            label_counts = Counter(harmonized_train["harmonized_label"])
            logger.info(f"Распределение гармонизированных меток в train ({name}):")
            for label, count in sorted(label_counts.items()):
                logger.info(f"  {label}: {count} ({count/len(harmonized_train)*100:.1f}%)")
    
    return harmonized_datasets


def merge_datasets(harmonized_datasets, split="train"):
    """Объединение датасетов для указанного сплита."""
    logger.info(f"Объединение датасетов для сплита {split}...")
    
    datasets_to_merge = []
    
    for name, dataset in harmonized_datasets.items():
        if split in dataset and dataset[split] is not None:
            datasets_to_merge.append(dataset[split])
    
    if not datasets_to_merge:
        logger.warning(f"Нет датасетов для объединения в сплите {split}")
        return None
    
    # Объединение датасетов
    merged_dataset = concatenate_datasets(datasets_to_merge)
    logger.info(f"Объединенный датасет для сплита {split}: {len(merged_dataset)} примеров")
    
    # Вывод статистики
    log_dataset_statistics(merged_dataset, split)
    
    return merged_dataset


def log_dataset_statistics(dataset, split):
    """Вывод статистики по датасету."""
    # Статистика по источникам
    source_counts = Counter(dataset["source"])
    logger.info(f"Распределение источников в объединенном датасете ({split}):")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count} ({count/len(dataset)*100:.1f}%)")
    
    # Статистика по меткам
    label_counts = Counter(dataset["harmonized_label"])
    logger.info(f"Распределение меток в объединенном датасете ({split}):")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count} ({count/len(dataset)*100:.1f}%)")


def count_examples_by_source(merged_train, merged_val, merged_test):
    """Подсчет количества примеров по источникам."""
    sources = {}
    
    for source in ["ag_news", "bbc_news", "20newsgroups"]:
        train_count = len([ex for ex in merged_train if ex["source"] == source]) if merged_train else 0
        val_count = len([ex for ex in merged_val if ex["source"] == source]) if merged_val else 0
        test_count = len([ex for ex in merged_test if ex["source"] == source]) if merged_test else 0
        sources[source] = train_count + val_count + test_count
    
    return sources


def create_validation_split(train_dataset, val_size=0.15):
    """Создание валидационной выборки из обучающей выборки.
    
    Args:
        train_dataset: Обучающая выборка
        val_size: Размер валидационной выборки (доля от обучающей выборки)
        
    Returns:
        tuple: (новая обучающая выборка, валидационная выборка)
    """
    logger.info(f"Создание валидационной выборки (размер: {val_size*100}%)...")
    
    # Преобразование в pandas DataFrame для удобства работы
    train_df = train_dataset.to_pandas()
    
    # Создание комбинированной метки для стратификации (класс + источник)
    train_df["strat_label"] = train_df["harmonized_label"] + "_" + train_df["source"]
    
    # Разделение на обучающую и валидационную выборки с сохранением стратификации
    new_train_df, val_df = train_test_split(
        train_df, 
        test_size=val_size, 
        random_state=42, 
        stratify=train_df["strat_label"]
    )
    
    # Удаление вспомогательной колонки
    new_train_df = new_train_df.drop(columns=["strat_label"])
    val_df = val_df.drop(columns=["strat_label"])
    
    # Сброс индексов перед преобразованием в Dataset
    new_train_df = new_train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Преобразование обратно в Dataset
    new_train_dataset = Dataset.from_pandas(new_train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    logger.info(f"Новая обучающая выборка: {len(new_train_dataset)} примеров")
    logger.info(f"Валидационная выборка: {len(val_dataset)} примеров")
    
    return new_train_dataset, val_dataset


def create_merged_dataset(harmonized_datasets):
    """Создание объединенного датасета."""
    logger.info(f"Создание объединенного датасета...")
    
    # Объединение train и test сплитов
    merged_train = merge_datasets(harmonized_datasets, "train")
    merged_test = merge_datasets(harmonized_datasets, "test")
    
    if merged_train is None or merged_test is None:
        logger.error("Не удалось создать объединенный датасет")
        return None
    
    # Создание валидационной выборки из обучающей
    merged_train, merged_val = create_validation_split(merged_train)
    
    # Создание DatasetDict
    merged_dataset = DatasetDict({
        "train": merged_train,
        "validation": merged_val,
        "test": merged_test
    })
    
    # Сохранение объединенного датасета
    output_dir = MERGED_DATA_DIR / "merged"
    merged_dataset.save_to_disk(str(output_dir))
    
    logger.info(f"Объединенный датасет сохранен в {output_dir}")
    
    # Сохранение метаданных
    metadata = {
        "name": "merged",
        "description": "Объединенный датасет из AG News, BBC News и 20 Newsgroups",
        "num_samples": {
            "train": len(merged_train),
            "validation": len(merged_val),
            "test": len(merged_test),
            "total": len(merged_train) + len(merged_val) + len(merged_test)
        },
        "label_mapping": "6 классов",
        "sources": count_examples_by_source(merged_train, merged_val, merged_test)
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return merged_dataset


def main():
    """Основная функция для объединения датасетов."""
    logger.info("Начало объединения датасетов...")
    
    # Загрузка очищенных датасетов
    datasets = load_clean_datasets()
    
    if not datasets:
        logger.error("Не удалось загрузить ни один датасет")
        return
    
    # Гармонизация меток
    harmonized_datasets = harmonize_labels(datasets)
    
    # Создание объединенного датасета
    merged_dataset = create_merged_dataset(harmonized_datasets)
    
    if merged_dataset:
        logger.info(f"Объединенный датасет успешно создан")
    
    logger.info("Объединение датасетов завершено")


if __name__ == "__main__":
    main() 