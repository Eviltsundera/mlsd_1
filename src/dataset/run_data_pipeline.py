#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для запуска полного процесса загрузки и анализа данных:
1. Загрузка датасетов
2. Анализ датасетов
"""

import os
import logging
import subprocess
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути к директориям
ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "src" / "data"

def run_script(script_name):
    """Запуск скрипта Python."""
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"Скрипт {script_path} не найден")
        return False
    
    logger.info(f"Запуск скрипта {script_name}...")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Скрипт {script_name} успешно выполнен")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при выполнении скрипта {script_name}: {e}")
        logger.error(f"Вывод: {e.stdout}")
        logger.error(f"Ошибки: {e.stderr}")
        return False

def main():
    """Основная функция для запуска процесса загрузки и анализа данных."""
    logger.info("Начало процесса загрузки и анализа данных...")
    
    # Создание необходимых директорий
    dirs = [
        ROOT_DIR / "data" / "raw",
        ROOT_DIR / "data" / "processed",
        ROOT_DIR / "models",
        ROOT_DIR / "notebooks" / "figures",
        ROOT_DIR / "src" / "data",
        ROOT_DIR / "src" / "features",
        ROOT_DIR / "src" / "models",
        ROOT_DIR / "src" / "visualization"
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Директория {directory} создана (или уже существует)")
    
    # Шаг 1: Загрузка датасетов
    if run_script("download_datasets.py"):
        logger.info("Загрузка датасетов успешно завершена")
    else:
        logger.error("Ошибка при загрузке датасетов")
        return
    
    # Шаг 2: Анализ датасетов
    if run_script("analyze_datasets.py"):
        logger.info("Анализ датасетов успешно завершен")
    else:
        logger.error("Ошибка при анализе датасетов")
        return
    
    logger.info("Процесс загрузки и анализа данных успешно завершен")

if __name__ == "__main__":
    main() 