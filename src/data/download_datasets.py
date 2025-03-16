"""
Скрипт для загрузки и подготовки датасетов новостных статей:
1. AG News (через Hugging Face)
2. BBC News (через Kaggle API)
3. 20 Newsgroups (через scikit-learn)
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import subprocess
import zipfile
import shutil

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

# Создаем директории, если они не существуют
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ag_news():
    """Загрузка датасета AG News через Hugging Face."""
    logger.info("Загрузка AG News датасета...")
    
    # Загрузка датасета
    ag_news = load_dataset("ag_news")
    
    # Сохранение в формате Arrow
    ag_news_dir = RAW_DATA_DIR / "ag_news"
    ag_news_dir.mkdir(exist_ok=True)
    
    ag_news.save_to_disk(str(ag_news_dir))
    
    # Сохранение метаданных
    metadata = {
        "name": "AG News",
        "description": "AG News - набор новостных статей с 4 категориями: World, Sports, Business, Sci/Tech",
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "num_samples": {
            "train": len(ag_news["train"]),
            "test": len(ag_news["test"])
        },
        "source": "Hugging Face Datasets",
        "license": "CC BY-SA 3.0"
    }
    
    with open(ag_news_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"AG News датасет сохранен в {ag_news_dir}")
    logger.info(f"Количество примеров: {len(ag_news['train'])} (train), {len(ag_news['test'])} (test)")
    
    return ag_news


def download_bbc_news():
    """Загрузка датасета BBC News через Kaggle API."""
    logger.info("Загрузка BBC News датасета...")
    
    bbc_news_dir = RAW_DATA_DIR / "bbc_news"
    bbc_news_dir.mkdir(exist_ok=True)
    
    # Проверка наличия файла kaggle.json
    kaggle_dir = Path.home() / ".kaggle"
    if not (kaggle_dir / "kaggle.json").exists():
        logger.warning("Файл kaggle.json не найден. Создаем директорию ~/.kaggle/")
        kaggle_dir.mkdir(exist_ok=True)
        
        # Запрос API ключа у пользователя
        logger.info("Для загрузки BBC News датасета требуется Kaggle API ключ.")
        logger.info("Получите ключ на странице https://www.kaggle.com/account")
        logger.info("Затем создайте файл ~/.kaggle/kaggle.json со следующим содержимым:")
        logger.info('{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}')
        
        # Проверка установки kaggle CLI
        try:
            subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Kaggle CLI не установлен. Устанавливаем...")
            subprocess.run(["pip", "install", "kaggle"], check=True)
        
        input("Нажмите Enter после создания файла kaggle.json...")
    
    # Установка прав доступа для kaggle.json
    try:
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
    except Exception as e:
        logger.warning(f"Не удалось установить права доступа для kaggle.json: {e}")
    
    # Загрузка датасета
    dataset_path = bbc_news_dir / "bbcnewsarchive.zip"
    if not dataset_path.exists():
        logger.info("Загрузка BBC News датасета с Kaggle...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "hgultekin/bbcnewsarchive", "-p", str(bbc_news_dir)],
                check=True
            )
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета с Kaggle: {e}")
            raise
    
    # Распаковка архива
    extract_dir = bbc_news_dir / "extracted"
    csv_path = extract_dir / "bbc-news-data.csv"
    
    if not csv_path.exists():
        logger.info("Распаковка архива...")
        extract_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Архив успешно распакован в {extract_dir}")
        except Exception as e:
            logger.error(f"Ошибка при распаковке архива: {e}")
            raise
    else:
        logger.info(f"Файл {csv_path} уже существует, пропускаем распаковку")
    
    # Проверка наличия CSV файла
    if not csv_path.exists():
        logger.error(f"Файл {csv_path} не найден после распаковки")
        raise FileNotFoundError(f"Файл {csv_path} не найден")
    
    # Чтение CSV файла
    logger.info(f"Чтение файла {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Загружено {len(df)} записей из CSV файла")
    
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
    
    return bbc_news


def download_20newsgroups():
    """Загрузка датасета 20 Newsgroups через scikit-learn."""
    logger.info("Загрузка 20 Newsgroups датасета...")
    
    newsgroups_dir = RAW_DATA_DIR / "20newsgroups"
    newsgroups_dir.mkdir(exist_ok=True)
    
    # Загрузка обучающего набора
    train_data = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Загрузка тестового набора
    test_data = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Преобразование в DatasetDict
    train_dataset = Dataset.from_dict({
        "text": train_data.data,
        "label": train_data.target,
        "label_name": [train_data.target_names[i] for i in train_data.target]
    })
    
    test_dataset = Dataset.from_dict({
        "text": test_data.data,
        "label": test_data.target,
        "label_name": [test_data.target_names[i] for i in test_data.target]
    })
    
    newsgroups = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Сохранение в формате Arrow
    newsgroups.save_to_disk(str(newsgroups_dir / "dataset"))
    
    # Сохранение метаданных
    metadata = {
        "name": "20 Newsgroups",
        "description": "20 Newsgroups - набор новостных статей с 20 категориями",
        "classes": train_data.target_names,
        "num_samples": {
            "train": len(train_data.data),
            "test": len(test_data.data),
            "total": len(train_data.data) + len(test_data.data)
        },
        "source": "scikit-learn",
        "license": "Свободное использование"
    }
    
    with open(newsgroups_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"20 Newsgroups датасет сохранен в {newsgroups_dir}")
    logger.info(f"Количество примеров: {len(train_data.data)} (train), {len(test_data.data)} (test)")
    
    return newsgroups


def main():
    """Основная функция для загрузки всех датасетов."""
    logger.info("Начало загрузки датасетов...")
    
    # Загрузка AG News
    try:
        ag_news = download_ag_news()
        logger.info("AG News успешно загружен")
    except Exception as e:
        logger.error(f"Ошибка при загрузке AG News: {e}")
    
    # Загрузка BBC News
    try:
        bbc_news = download_bbc_news()
        logger.info("BBC News успешно загружен")
    except Exception as e:
        logger.error(f"Ошибка при загрузке BBC News: {e}")
    
    # Загрузка 20 Newsgroups
    try:
        newsgroups = download_20newsgroups()
        logger.info("20 Newsgroups успешно загружен")
    except Exception as e:
        logger.error(f"Ошибка при загрузке 20 Newsgroups: {e}")
    
    logger.info("Загрузка датасетов завершена")


if __name__ == "__main__":
    main() 