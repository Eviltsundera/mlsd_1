#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для очистки датасетов:
- Удаление дубликатов в обучающей и тестовой выборках
- Удаление пересечений между обучающей и тестовой выборками
- Сохранение очищенных датасетов
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from collections import Counter
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути к директориям
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# Создаем директорию для очищенных датасетов
CLEAN_DATA_DIR = PROCESSED_DATA_DIR / "clean"
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Загрузка всех доступных датасетов."""
    datasets = {}
    
    # AG News
    ag_news_path = RAW_DATA_DIR / "ag_news"
    if ag_news_path.exists():
        logger.info("Загрузка AG News датасета...")
        try:
            datasets["ag_news"] = load_from_disk(str(ag_news_path))
            logger.info("AG News датасет успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка при загрузке AG News: {e}")
    
    # BBC News
    bbc_news_path = RAW_DATA_DIR / "bbc_news" / "dataset"
    if bbc_news_path.exists():
        logger.info("Загрузка BBC News датасета...")
        try:
            datasets["bbc_news"] = load_from_disk(str(bbc_news_path))
            logger.info("BBC News датасет успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка при загрузке BBC News: {e}")
    
    # 20 Newsgroups
    newsgroups_path = RAW_DATA_DIR / "20newsgroups" / "dataset"
    if newsgroups_path.exists():
        logger.info("Загрузка 20 Newsgroups датасета...")
        try:
            datasets["20newsgroups"] = load_from_disk(str(newsgroups_path))
            logger.info("20 Newsgroups датасет успешно загружен")
        except Exception as e:
            logger.error(f"Ошибка при загрузке 20 Newsgroups: {e}")
    
    return datasets


def normalize_text(text):
    """Нормализация текста для сравнения."""
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации и специальных символов
    text = re.sub(r'[^\w\s]', '', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_duplicates(dataset, text_col="text"):
    """Удаление дубликатов из датасета."""
    logger.info(f"Удаление дубликатов (колонка: {text_col})...")
    
    # Обработка обучающей выборки
    if "train" in dataset:
        train_texts = dataset["train"][text_col]
        train_normalized = [normalize_text(text) for text in train_texts]
        
        # Поиск дубликатов
        seen = set()
        duplicates = []
        
        for i, text in enumerate(train_normalized):
            if text in seen:
                duplicates.append(i)
            else:
                seen.add(text)
        
        logger.info(f"Найдено {len(duplicates)} дубликатов в обучающей выборке")
        
        # Удаление дубликатов
        if duplicates:
            keep_indices = [i for i in range(len(train_texts)) if i not in duplicates]
            dataset["train"] = dataset["train"].select(keep_indices)
            logger.info(f"Обучающая выборка после удаления дубликатов: {len(dataset['train'])} примеров")
    
    # Обработка тестовой выборки
    if "test" in dataset:
        test_texts = dataset["test"][text_col]
        test_normalized = [normalize_text(text) for text in test_texts]
        
        # Поиск дубликатов
        seen = set()
        duplicates = []
        
        for i, text in enumerate(test_normalized):
            if text in seen:
                duplicates.append(i)
            else:
                seen.add(text)
        
        logger.info(f"Найдено {len(duplicates)} дубликатов в тестовой выборке")
        
        # Удаление дубликатов
        if duplicates:
            keep_indices = [i for i in range(len(test_texts)) if i not in duplicates]
            dataset["test"] = dataset["test"].select(keep_indices)
            logger.info(f"Тестовая выборка после удаления дубликатов: {len(dataset['test'])} примеров")
    
    return dataset


def remove_train_test_overlap(dataset, text_col="text"):
    """Удаление пересечений между обучающей и тестовой выборками."""
    logger.info(f"Удаление пересечений между train и test (колонка: {text_col})...")
    
    if "train" not in dataset or "test" not in dataset:
        logger.warning("Датасет не содержит разделения на train/test")
        return dataset
    
    # Нормализация текстов
    train_texts = dataset["train"][text_col]
    test_texts = dataset["test"][text_col]
    
    train_normalized = [normalize_text(text) for text in train_texts]
    test_normalized = [normalize_text(text) for text in test_texts]
    
    # Поиск пересечений
    train_set = set(train_normalized)
    test_set = set(test_normalized)
    
    intersection = train_set.intersection(test_set)
    logger.info(f"Найдено {len(intersection)} пересечений между train и test")
    
    if not intersection:
        return dataset
    
    # Удаление пересечений из тестовой выборки
    overlap_indices = [i for i, text in enumerate(test_normalized) if text in train_set]
    
    if overlap_indices:
        keep_indices = [i for i in range(len(test_texts)) if i not in overlap_indices]
        dataset["test"] = dataset["test"].select(keep_indices)
        logger.info(f"Тестовая выборка после удаления пересечений: {len(dataset['test'])} примеров")
    
    return dataset


def clean_dataset(dataset, name, text_col="text"):
    """Очистка датасета: удаление дубликатов и пересечений."""
    logger.info(f"Очистка датасета {name}...")
    
    # Определение колонки с текстом
    if name == "bbc_news":
        text_col = "content"
    
    # Удаление дубликатов
    clean_dataset = remove_duplicates(dataset, text_col)
    
    # Удаление пересечений между train и test
    clean_dataset = remove_train_test_overlap(clean_dataset, text_col)
    
    # Сохранение очищенного датасета
    output_dir = CLEAN_DATA_DIR / name
    clean_dataset.save_to_disk(str(output_dir))
    
    logger.info(f"Очищенный датасет {name} сохранен в {output_dir}")
    
    # Сохранение метаданных
    if "train" in clean_dataset and "test" in clean_dataset:
        metadata = {
            "name": name,
            "num_samples": {
                "train": len(clean_dataset["train"]),
                "test": len(clean_dataset["test"]),
                "total": len(clean_dataset["train"]) + len(clean_dataset["test"])
            },
            "cleaning_applied": {
                "duplicates_removed": True,
                "train_test_overlap_removed": True
            }
        }
        
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return clean_dataset


def main():
    """Основная функция для очистки всех датасетов."""
    logger.info("Начало очистки датасетов...")
    
    # Загрузка датасетов
    datasets = load_datasets()
    
    if not datasets:
        logger.error("Не удалось загрузить ни один датасет")
        return
    
    # Очистка каждого датасета
    clean_datasets = {}
    
    for name, dataset in datasets.items():
        clean_datasets[name] = clean_dataset(dataset, name)
    
    # Сохранение сводной информации
    summary = {
        "datasets": {},
        "total_samples": {
            "original": {},
            "cleaned": {}
        }
    }
    
    for name, dataset in clean_datasets.items():
        if "train" in dataset and "test" in dataset:
            original_train = len(datasets[name]["train"])
            original_test = len(datasets[name]["test"])
            cleaned_train = len(dataset["train"])
            cleaned_test = len(dataset["test"])
            
            summary["datasets"][name] = {
                "original": {
                    "train": original_train,
                    "test": original_test,
                    "total": original_train + original_test
                },
                "cleaned": {
                    "train": cleaned_train,
                    "test": cleaned_test,
                    "total": cleaned_train + cleaned_test
                },
                "removed": {
                    "train": original_train - cleaned_train,
                    "test": original_test - cleaned_test,
                    "total": (original_train + original_test) - (cleaned_train + cleaned_test)
                }
            }
            
            summary["total_samples"]["original"][name] = original_train + original_test
            summary["total_samples"]["cleaned"][name] = cleaned_train + cleaned_test
    
    summary["total_samples"]["original"]["total"] = sum(summary["total_samples"]["original"].values())
    summary["total_samples"]["cleaned"]["total"] = sum(summary["total_samples"]["cleaned"].values())
    
    with open(CLEAN_DATA_DIR / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Сводная информация сохранена в {CLEAN_DATA_DIR / 'cleaning_summary.json'}")
    logger.info("Очистка датасетов завершена")


if __name__ == "__main__":
    main() 