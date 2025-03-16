#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для анализа объединенного датасета:
- Загрузка объединенного датасета
- Анализ распределения классов
- Анализ длины текстов
- Анализ распределения источников данных
- Сохранение результатов анализа
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_from_disk
from collections import Counter
import seaborn as sns
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути к директориям
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MERGED_DATA_DIR = PROCESSED_DATA_DIR / "merged"
ANALYSIS_DIR = PROCESSED_DATA_DIR / "analysis"

# Создаем директорию для результатов анализа
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_merged_dataset():
    """Загрузка объединенного датасета."""
    dataset_path = MERGED_DATA_DIR / "merged"
    
    if dataset_path.exists():
        logger.info("Загрузка объединенного датасета...")
        try:
            dataset = load_from_disk(str(dataset_path))
            logger.info("Датасет успешно загружен")
            return dataset
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {e}")
    else:
        logger.error(f"Путь к датасету не существует: {dataset_path}")
    
    return None


def get_text_from_row(row):
    """Получение текста из строки датасета в зависимости от источника."""
    source = row["source"]
    text = None
    
    if source == "ag_news":
        # Для AG News текст находится в колонке "text"
        text = row.get("text", "")
    elif source == "bbc_news":
        # Для BBC News текст находится в колонке "content"
        text = row.get("content", "")
    elif source == "20newsgroups":
        # Для 20 Newsgroups текст находится в колонке "text"
        text = row.get("text", "")
    else:
        logger.warning(f"Неизвестный источник: {source}")
        text = ""
    
    # Проверка на None и пустые строки
    if text is None:
        text = ""
    
    return text


def create_and_save_plot(data, x, y, title, xlabel, ylabel, output_path, 
                         plot_type="bar", hue=None, rotation=0, ylim=None, figsize=(10, 6)):
    """Создание и сохранение визуализации."""
    plt.figure(figsize=figsize)
    
    if plot_type == "bar":
        sns.barplot(x=x, y=y, data=data, hue=hue)
    elif plot_type == "hist":
        sns.histplot(data=data, x=x, hue=hue, bins=50, kde=True)
    elif plot_type == "box":
        sns.boxplot(x=x, y=y, data=data, hue=hue)
    elif plot_type == "heatmap":
        sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if rotation != 0:
        plt.xticks(rotation=rotation)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Визуализация сохранена в {output_path}")


def analyze_class_distribution(dataset, split="train"):
    """Анализ распределения классов в датасете."""
    logger.info(f"Анализ распределения классов в датасете (сплит {split})...")
    
    if split not in dataset:
        logger.warning(f"Сплит {split} отсутствует в датасете")
        return None
    
    # Получение данных
    data = dataset[split].to_pandas()
    
    # Анализ распределения гармонизированных меток
    label_counts = Counter(data["harmonized_label"])
    total = len(data)
    
    # Создание DataFrame с результатами
    results = pd.DataFrame({
        "label": list(label_counts.keys()),
        "count": list(label_counts.values()),
        "percentage": [count / total * 100 for count in label_counts.values()]
    })
    
    # Сортировка по количеству
    results = results.sort_values("count", ascending=False).reset_index(drop=True)
    
    # Вывод результатов
    logger.info(f"Распределение классов в датасете (сплит {split}):")
    for _, row in results.iterrows():
        logger.info(f"  {row['label']}: {row['count']} ({row['percentage']:.1f}%)")
    
    # Создание и сохранение визуализации
    output_path = ANALYSIS_DIR / f"class_distribution_{split}.png"
    create_and_save_plot(
        data=results,
        x="label",
        y="percentage",
        title=f"Распределение классов в датасете (сплит {split})",
        xlabel="Класс",
        ylabel="Процент (%)",
        output_path=output_path,
        rotation=45
    )
    
    return results


def analyze_text_length(dataset, split="train"):
    """Анализ длины текстов в датасете."""
    logger.info(f"Анализ длины текстов в датасете (сплит {split})...")
    
    if split not in dataset:
        logger.warning(f"Сплит {split} отсутствует в датасете")
        return None
    
    # Получение данных
    data = dataset[split].to_pandas()
    
    # Определение колонки с текстом в зависимости от источника и подсчет длины
    text_lengths = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Подсчет длины текстов"):
        text = get_text_from_row(row)
        if not text:
            logger.warning(f"Обнаружен пустой текст в строке {idx}")
        
        # Подсчет длины текста (в словах)
        text_lengths.append(len(text.split()))
    
    # Добавление длины текста в DataFrame
    data["text_length"] = text_lengths
    
    # Статистика по длине текстов
    stats = {
        "min": np.min(text_lengths),
        "max": np.max(text_lengths),
        "mean": np.mean(text_lengths),
        "median": np.median(text_lengths),
        "std": np.std(text_lengths),
        "q1": np.percentile(text_lengths, 25),
        "q3": np.percentile(text_lengths, 75)
    }
    
    # Вывод статистики
    logger.info(f"Статистика длины текстов в датасете (сплит {split}):")
    logger.info(f"  Минимум: {stats['min']}")
    logger.info(f"  Максимум: {stats['max']}")
    logger.info(f"  Среднее: {stats['mean']:.1f}")
    logger.info(f"  Медиана: {stats['median']}")
    logger.info(f"  Стандартное отклонение: {stats['std']:.1f}")
    logger.info(f"  Q1 (25%): {stats['q1']}")
    logger.info(f"  Q3 (75%): {stats['q3']}")
    
    # Статистика по длине текстов в зависимости от источника
    source_stats = data.groupby("source")["text_length"].agg(["min", "max", "mean", "median", "std"])
    
    # Вывод статистики по источникам
    logger.info(f"Статистика длины текстов по источникам в датасете (сплит {split}):")
    for source, row in source_stats.iterrows():
        logger.info(f"  {source}:")
        logger.info(f"    Минимум: {row['min']}")
        logger.info(f"    Максимум: {row['max']}")
        logger.info(f"    Среднее: {row['mean']:.1f}")
        logger.info(f"    Медиана: {row['median']}")
        logger.info(f"    Стандартное отклонение: {row['std']:.1f}")
    
    # Создание и сохранение визуализации распределения длины текстов
    output_path = ANALYSIS_DIR / f"text_length_distribution_{split}.png"
    create_and_save_plot(
        data=data,
        x="text_length",
        y=None,
        title=f"Распределение длины текстов в датасете (сплит {split})",
        xlabel="Длина текста (слов)",
        ylabel="Количество",
        output_path=output_path,
        plot_type="hist",
        hue="source",
        figsize=(12, 6),
        ylim=None
    )
    
    # Создание и сохранение визуализации длины текстов по классам
    output_path = ANALYSIS_DIR / f"text_length_by_class_{split}.png"
    create_and_save_plot(
        data=data,
        x="harmonized_label",
        y="text_length",
        title=f"Длина текстов по классам в датасете (сплит {split})",
        xlabel="Класс",
        ylabel="Длина текста (слов)",
        output_path=output_path,
        plot_type="box",
        figsize=(12, 6),
        ylim=(0, min(1000, stats["max"]))
    )
    
    return stats, source_stats


def analyze_source_distribution(dataset, split="train"):
    """Анализ распределения источников данных в датасете."""
    logger.info(f"Анализ распределения источников в датасете (сплит {split})...")
    
    if split not in dataset:
        logger.warning(f"Сплит {split} отсутствует в датасете")
        return None
    
    # Получение данных
    data = dataset[split].to_pandas()
    
    # Анализ распределения источников
    source_counts = Counter(data["source"])
    total = len(data)
    
    # Создание DataFrame с результатами
    results = pd.DataFrame({
        "source": list(source_counts.keys()),
        "count": list(source_counts.values()),
        "percentage": [count / total * 100 for count in source_counts.values()]
    })
    
    # Сортировка по количеству
    results = results.sort_values("count", ascending=False).reset_index(drop=True)
    
    # Вывод результатов
    logger.info(f"Распределение источников в датасете (сплит {split}):")
    for _, row in results.iterrows():
        logger.info(f"  {row['source']}: {row['count']} ({row['percentage']:.1f}%)")
    
    # Создание и сохранение визуализации
    output_path = ANALYSIS_DIR / f"source_distribution_{split}.png"
    create_and_save_plot(
        data=results,
        x="source",
        y="percentage",
        title=f"Распределение источников в датасете (сплит {split})",
        xlabel="Источник",
        ylabel="Процент (%)",
        output_path=output_path
    )
    
    # Анализ распределения классов по источникам
    source_class_distribution = {}
    for source in source_counts.keys():
        source_data = data[data["source"] == source]
        label_counts = Counter(source_data["harmonized_label"])
        source_total = len(source_data)
        
        source_class_distribution[source] = {
            "counts": {label: count for label, count in label_counts.items()},
            "percentages": {label: count / source_total * 100 for label, count in label_counts.items()}
        }
    
    # Вывод результатов
    logger.info(f"Распределение классов по источникам в датасете (сплит {split}):")
    for source, distribution in source_class_distribution.items():
        logger.info(f"  {source}:")
        for label, percentage in sorted(distribution["percentages"].items(), key=lambda x: x[1], reverse=True):
            count = distribution["counts"][label]
            logger.info(f"    {label}: {count} ({percentage:.1f}%)")
    
    # Создание визуализации распределения классов по источникам
    # Подготовка данных для визуализации
    viz_data = []
    for source, distribution in source_class_distribution.items():
        for label, percentage in distribution["percentages"].items():
            viz_data.append({
                "source": source,
                "label": label,
                "percentage": percentage
            })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Создание тепловой карты
    pivot_table = viz_df.pivot(index="source", columns="label", values="percentage")
    
    # Сохранение визуализации
    output_path = ANALYSIS_DIR / f"class_by_source_{split}.png"
    create_and_save_plot(
        data=pivot_table,
        x=None,
        y=None,
        title=f"Распределение классов по источникам в датасете (сплит {split})",
        xlabel="",
        ylabel="",
        output_path=output_path,
        plot_type="heatmap",
        figsize=(15, 8)
    )
    
    return results, source_class_distribution


def convert_numpy_types(obj):
    """Преобразование numpy типов в стандартные Python типы для JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_analysis_results(results):
    """Сохранение результатов анализа в JSON-файл."""
    logger.info("Сохранение результатов анализа датасета...")
    
    output_path = ANALYSIS_DIR / "analysis_results.json"
    
    # Преобразование результатов в формат, подходящий для JSON
    json_results = {}
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            json_results[key] = value.to_dict(orient="records")
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict) and isinstance(value[1], pd.DataFrame):
            json_results[key] = {
                "stats": convert_numpy_types(value[0]),
                "source_stats": value[1].to_dict(orient="index")
            }
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], pd.DataFrame) and isinstance(value[1], dict):
            json_results[key] = {
                "source_distribution": value[0].to_dict(orient="records"),
                "class_by_source": convert_numpy_types(value[1])
            }
        else:
            json_results[key] = convert_numpy_types(value)
    
    # Сохранение результатов
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты анализа сохранены в {output_path}")


def create_comparison_plots(dataset):
    """Создание сравнительных визуализаций для всех сплитов."""
    logger.info("Создание сравнительных визуализаций для всех сплитов...")
    
    # Проверка наличия всех сплитов
    splits = ["train", "validation", "test"]
    if not all(split in dataset for split in splits):
        logger.warning("Не все сплиты присутствуют в датасете")
        return
    
    # Сравнение распределения классов
    class_distribution = {}
    for split in splits:
        data = dataset[split].to_pandas()
        label_counts = Counter(data["harmonized_label"])
        total = len(data)
        class_distribution[split] = {label: count/total*100 for label, count in label_counts.items()}
    
    # Создание DataFrame для визуализации
    viz_data = []
    for split, distribution in class_distribution.items():
        for label, percentage in distribution.items():
            viz_data.append({
                "split": split,
                "label": label,
                "percentage": percentage
            })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Создание и сохранение визуализации
    plt.figure(figsize=(14, 8))
    sns.barplot(x="label", y="percentage", hue="split", data=viz_df)
    plt.title("Сравнение распределения классов между сплитами")
    plt.xlabel("Класс")
    plt.ylabel("Процент (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = ANALYSIS_DIR / "class_distribution_comparison.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Визуализация сохранена в {output_path}")
    
    # Сравнение распределения источников
    source_distribution = {}
    for split in splits:
        data = dataset[split].to_pandas()
        source_counts = Counter(data["source"])
        total = len(data)
        source_distribution[split] = {source: count/total*100 for source, count in source_counts.items()}
    
    # Создание DataFrame для визуализации
    viz_data = []
    for split, distribution in source_distribution.items():
        for source, percentage in distribution.items():
            viz_data.append({
                "split": split,
                "source": source,
                "percentage": percentage
            })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Создание и сохранение визуализации
    plt.figure(figsize=(12, 8))
    sns.barplot(x="source", y="percentage", hue="split", data=viz_df)
    plt.title("Сравнение распределения источников между сплитами")
    plt.xlabel("Источник")
    plt.ylabel("Процент (%)")
    plt.tight_layout()
    output_path = ANALYSIS_DIR / "source_distribution_comparison.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Визуализация сохранена в {output_path}")
    
    # Сравнение длины текстов
    text_length_stats = {}
    for split in splits:
        data = dataset[split].to_pandas()
        text_lengths = []
        for _, row in data.iterrows():
            text = get_text_from_row(row)
            text_lengths.append(len(text.split()))
        
        text_length_stats[split] = {
            "mean": np.mean(text_lengths),
            "median": np.median(text_lengths),
            "std": np.std(text_lengths)
        }
    
    # Создание DataFrame для визуализации
    viz_data = []
    for split, stats in text_length_stats.items():
        for stat_name, value in stats.items():
            viz_data.append({
                "split": split,
                "stat": stat_name,
                "value": value
            })
    
    viz_df = pd.DataFrame(viz_data)
    
    # Создание и сохранение визуализации
    plt.figure(figsize=(10, 6))
    sns.barplot(x="stat", y="value", hue="split", data=viz_df)
    plt.title("Сравнение статистики длины текстов между сплитами")
    plt.xlabel("Статистика")
    plt.ylabel("Значение")
    plt.tight_layout()
    output_path = ANALYSIS_DIR / "text_length_stats_comparison.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Визуализация сохранена в {output_path}")
    
    return {
        "class_distribution": class_distribution,
        "source_distribution": source_distribution,
        "text_length_stats": text_length_stats
    }


def main():
    """Основная функция для анализа объединенного датасета."""
    logger.info("Начало анализа объединенного датасета...")
    
    # Загрузка объединенного датасета
    dataset = load_merged_dataset()
    
    if dataset is None:
        logger.error("Не удалось загрузить датасет")
        return
    
    results = {}
    
    # Анализ для каждого сплита
    for split in ["train", "validation", "test"]:
        if split in dataset:
            # Анализ распределения классов
            results[f"class_distribution_{split}"] = analyze_class_distribution(dataset, split)
            
            # Анализ длины текстов
            results[f"text_length_{split}"] = analyze_text_length(dataset, split)
            
            # Анализ распределения источников
            results[f"source_distribution_{split}"] = analyze_source_distribution(dataset, split)
    
    # Создание сравнительных визуализаций
    results["comparison"] = create_comparison_plots(dataset)
    
    # Сохранение результатов анализа
    save_analysis_results(results)
    
    logger.info("Анализ объединенного датасета завершен")


if __name__ == "__main__":
    main() 