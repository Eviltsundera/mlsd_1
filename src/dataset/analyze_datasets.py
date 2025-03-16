"""
Скрипт для анализа загруженных датасетов новостных статей:
- Распределение классов
- Статистика по длине текстов
- Проверка на дубликаты
- Поиск пересечений между датасетами
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
FIGURES_DIR = ROOT_DIR / "notebooks" / "figures"

# Создаем директории, если они не существуют
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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


def analyze_class_distribution(datasets):
    """Анализ распределения классов в датасетах."""
    logger.info("Анализ распределения классов...")
    
    # Создаем отдельный график для каждого датасета
    for name, dataset in datasets.items():
        plt.figure(figsize=(15, 6))
        
        # Определение колонки с метками
        label_col = "label"
        if name == "bbc_news":
            label_col = "category"
        
        # Подсчет количества примеров по классам
        if "train" in dataset:
            plt.subplot(1, 2, 1)
            train_counts = Counter(dataset["train"][label_col])
            labels, counts = zip(*sorted(train_counts.items()))
            
            if name == "20newsgroups":
                # Для 20 Newsgroups используем имена классов
                labels = [dataset["train"]["label_name"][dataset["train"]["label"].index(l)] for l in labels]
            
            plt.bar(labels, counts)
            plt.title(f"{name} - Распределение классов (Train)")
            plt.xticks(rotation=90)
            
            # Вывод процентного соотношения
            for i, count in enumerate(counts):
                percentage = 100 * count / sum(counts)
                plt.text(i, count, f"{percentage:.1f}%", ha='center', va='bottom')
        
        if "test" in dataset:
            plt.subplot(1, 2, 2)
            test_counts = Counter(dataset["test"][label_col])
            labels, counts = zip(*sorted(test_counts.items()))
            
            if name == "20newsgroups":
                # Для 20 Newsgroups используем имена классов
                labels = [dataset["test"]["label_name"][dataset["test"]["label"].index(l)] for l in labels]
            
            plt.bar(labels, counts)
            plt.title(f"{name} - Распределение классов (Test)")
            plt.xticks(rotation=90)
            
            # Вывод процентного соотношения
            for i, count in enumerate(counts):
                percentage = 100 * count / sum(counts)
                plt.text(i, count, f"{percentage:.1f}%", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"class_distribution_{name}.png")
        logger.info(f"График распределения классов для {name} сохранен в {FIGURES_DIR / f'class_distribution_{name}.png'}")
        plt.close()


def analyze_text_length(datasets):
    """Анализ длины текстов в датасетах."""
    logger.info("Анализ длины текстов...")
    
    # Создаем отдельный график для каждого датасета
    for name, dataset in datasets.items():
        plt.figure(figsize=(15, 6))
        
        # Определение колонки с текстом
        text_col = "text"
        if name == "ag_news":
            text_col = "text"
        elif name == "bbc_news":
            text_col = "content"
        
        # Подсчет длины текстов
        if "train" in dataset:
            plt.subplot(1, 2, 1)
            text_lengths = [len(text.split()) for text in dataset["train"][text_col]]
            
            sns.histplot(text_lengths, bins=50, kde=True)
            plt.title(f"{name} - Распределение длины текстов (Train)")
            plt.xlabel("Количество слов")
            plt.ylabel("Частота")
            
            # Вывод статистики
            plt.text(0.7, 0.8, 
                     f"Мин: {min(text_lengths)}\n"
                     f"Макс: {max(text_lengths)}\n"
                     f"Среднее: {np.mean(text_lengths):.1f}\n"
                     f"Медиана: {np.median(text_lengths):.1f}",
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
        
        if "test" in dataset:
            plt.subplot(1, 2, 2)
            text_lengths = [len(text.split()) for text in dataset["test"][text_col]]
            
            sns.histplot(text_lengths, bins=50, kde=True)
            plt.title(f"{name} - Распределение длины текстов (Test)")
            plt.xlabel("Количество слов")
            plt.ylabel("Частота")
            
            # Вывод статистики
            plt.text(0.7, 0.8, 
                     f"Мин: {min(text_lengths)}\n"
                     f"Макс: {max(text_lengths)}\n"
                     f"Среднее: {np.mean(text_lengths):.1f}\n"
                     f"Медиана: {np.median(text_lengths):.1f}",
                     transform=plt.gca().transAxes,
                     bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"text_length_distribution_{name}.png")
        logger.info(f"График распределения длины текстов для {name} сохранен в {FIGURES_DIR / f'text_length_distribution_{name}.png'}")
        plt.close()


def check_duplicates(datasets):
    """Проверка на дубликаты внутри датасетов."""
    logger.info("Проверка на дубликаты...")
    
    results = {}
    
    for name, dataset in datasets.items():
        logger.info(f"Проверка дубликатов в {name}...")
        
        # Определение колонки с текстом
        text_col = "text"
        if name == "bbc_news":
            text_col = "content"
        
        # Проверка дубликатов в train
        if "train" in dataset:
            train_texts = dataset["train"][text_col]
            train_duplicates = len(train_texts) - len(set(train_texts))
            logger.info(f"{name} (train): найдено {train_duplicates} дубликатов из {len(train_texts)} текстов")
        
        # Проверка дубликатов в test
        if "test" in dataset:
            test_texts = dataset["test"][text_col]
            test_duplicates = len(test_texts) - len(set(test_texts))
            logger.info(f"{name} (test): найдено {test_duplicates} дубликатов из {len(test_texts)} текстов")
        
        # Проверка пересечений между train и test
        if "train" in dataset and "test" in dataset:
            train_set = set(train_texts)
            test_set = set(test_texts)
            intersection = train_set.intersection(test_set)
            
            logger.info(f"{name}: найдено {len(intersection)} пересечений между train и test")
            
            results[name] = {
                "train_duplicates": train_duplicates,
                "test_duplicates": test_duplicates,
                "train_test_intersection": len(intersection)
            }
    
    return results


def find_dataset_intersections(datasets):
    """Поиск пересечений между разными датасетами."""
    logger.info("Поиск пересечений между датасетами...")
    
    # Извлечение текстов из всех датасетов
    dataset_texts = {}
    
    for name, dataset in datasets.items():
        # Определение колонки с текстом
        text_col = "text"
        if name == "bbc_news":
            text_col = "content"
        
        texts = []
        if "train" in dataset:
            texts.extend(dataset["train"][text_col])
        if "test" in dataset:
            texts.extend(dataset["test"][text_col])
        
        # Нормализация текстов (удаление пунктуации, приведение к нижнему регистру)
        normalized_texts = [re.sub(r'[^\w\s]', '', text.lower()) for text in texts]
        dataset_texts[name] = normalized_texts
    
    # Проверка точных совпадений
    intersections = {}
    
    for name1 in dataset_texts:
        for name2 in dataset_texts:
            if name1 >= name2:  # Избегаем дублирования (A∩B = B∩A)
                continue
            
            set1 = set(dataset_texts[name1])
            set2 = set(dataset_texts[name2])
            intersection = set1.intersection(set2)
            
            logger.info(f"Пересечение {name1} и {name2}: {len(intersection)} текстов")
            
            intersections[f"{name1}_vs_{name2}"] = len(intersection)
    
    return intersections


def main():
    """Основная функция для анализа датасетов."""
    logger.info("Начало анализа датасетов...")
    
    # Загрузка датасетов
    datasets = load_datasets()
    
    if not datasets:
        logger.error("Не удалось загрузить ни один датасет. Убедитесь, что скрипт download_datasets.py был выполнен.")
        return
    
    # Анализ распределения классов
    analyze_class_distribution(datasets)
    
    # Анализ длины текстов
    analyze_text_length(datasets)
    
    # Проверка на дубликаты
    duplicate_results = check_duplicates(datasets)
    
    # Поиск пересечений между датасетами
    intersection_results = find_dataset_intersections(datasets)
    
    # Сохранение результатов анализа
    results = {
        "duplicates": duplicate_results,
        "intersections": intersection_results
    }
    
    with open(PROCESSED_DATA_DIR / "dataset_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты анализа сохранены в {PROCESSED_DATA_DIR / 'dataset_analysis.json'}")
    logger.info("Анализ датасетов завершен")


if __name__ == "__main__":
    main() 