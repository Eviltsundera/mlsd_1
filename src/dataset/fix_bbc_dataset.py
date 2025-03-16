#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для исправления датасета BBC News:
- Преобразование CSV-файла в формат Arrow
- Сохранение в директории dataset
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

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

def fix_bbc_news():
    """Исправление датасета BBC News."""
    logger.info("Исправление датасета BBC News...")
    
    bbc_news_dir = RAW_DATA_DIR / "bbc_news"
    extract_dir = bbc_news_dir / "extracted"
    csv_path = extract_dir / "bbc-news-data.csv"
    
    # Проверка наличия CSV файла
    if not csv_path.exists():
        logger.error(f"Файл {csv_path} не найден")
        return False
    
    # Чтение CSV файла с обработкой ошибок
    logger.info(f"Чтение файла {csv_path}...")
    try:
        # Пробуем разные параметры для чтения CSV
        try:
            df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
        except TypeError:  # В новых версиях pandas параметры изменились
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        
        logger.info(f"Загружено {len(df)} записей из CSV файла")
        
        # Проверка наличия необходимых колонок
        required_columns = ["category", "content"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"В CSV-файле отсутствуют необходимые колонки: {missing_columns}")
            
            # Попробуем посмотреть первые строки файла для диагностики
            with open(csv_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                logger.info(f"Заголовок CSV: {header}")
                
                # Попробуем определить разделитель
                if ',' in header:
                    separator = ','
                elif ';' in header:
                    separator = ';'
                elif '\t' in header:
                    separator = '\t'
                else:
                    separator = None
                
                logger.info(f"Предполагаемый разделитель: {separator}")
                
                if separator:
                    df = pd.read_csv(csv_path, sep=separator, on_bad_lines='skip')
                    logger.info(f"Повторная загрузка с разделителем '{separator}': {len(df)} записей")
                    
                    # Проверка колонок снова
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        logger.error(f"Все еще отсутствуют колонки: {missing_columns}")
                        return False
        
        # Проверка на пустые значения
        if df["category"].isna().any() or df["content"].isna().any():
            logger.warning("Обнаружены пустые значения в колонках category или content")
            logger.info("Удаление строк с пустыми значениями...")
            df = df.dropna(subset=["category", "content"])
            logger.info(f"После удаления пустых значений: {len(df)} записей")
        
        # Разделение на train/test
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["category"]
        )
        
        # Преобразование в DatasetDict
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        bbc_news = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        # Сохранение в формате Arrow
        dataset_dir = bbc_news_dir / "dataset"
        bbc_news.save_to_disk(str(dataset_dir))
        logger.info(f"Датасет сохранен в {dataset_dir}")
        
        # Сохранение метаданных
        categories = df["category"].unique().tolist()
        metadata = {
            "name": "BBC News",
            "description": "BBC News - набор новостных статей с 5 категориями: business, entertainment, politics, sport, tech",
            "classes": categories,
            "num_samples": {
                "train": len(train_df),
                "test": len(test_df),
                "total": len(df)
            },
            "source": "Kaggle (hgultekin/bbcnewsarchive)",
            "license": "Требуется указание источника при публикации"
        }
        
        with open(bbc_news_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"BBC News датасет сохранен в {bbc_news_dir}")
        logger.info(f"Количество примеров: {len(train_df)} (train), {len(test_df)} (test)")
        
        return True
    
    except Exception as e:
        logger.error(f"Ошибка при обработке CSV-файла: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Основная функция."""
    logger.info("Начало исправления датасета BBC News...")
    
    if fix_bbc_news():
        logger.info("Исправление датасета BBC News успешно завершено")
    else:
        logger.error("Ошибка при исправлении датасета BBC News")
    
    logger.info("Исправление датасета BBC News завершено")

if __name__ == "__main__":
    main()