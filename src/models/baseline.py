#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Бейзлайн модель для классификации новостных статей.
Использует TF-IDF векторизацию и Logistic Regression.
"""

import os
import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути к данным и результатам
MERGED_DATA_DIR = "data/processed/merged/merged"
RESULTS_DIR = "models/baseline"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    """Загрузка объединенного датасета."""
    logger.info("Загрузка датасета...")
    dataset = load_from_disk(MERGED_DATA_DIR)
    return dataset

def preprocess_text(text):
    """Простая предобработка текста."""
    if isinstance(text, str):
        return text.lower()
    return ""

def train_baseline(train_data, val_data, test_data):
    """Обучение бейзлайна."""
    logger.info("Подготовка данных для обучения...")
    
    # Инициализация TF-IDF векторизатора
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        preprocessor=preprocess_text
    )
    
    # Векторизация текстов
    X_train = vectorizer.fit_transform(train_data['text'])
    X_val = vectorizer.transform(val_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    
    # Получение меток
    y_train = train_data['harmonized_label']
    y_val = val_data['harmonized_label']
    y_test = test_data['harmonized_label']
    
    # Обучение модели
    logger.info("Обучение модели...")
    model = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Оценка на валидационной выборке
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    logger.info(f"Валидационная accuracy: {val_accuracy:.4f}")
    logger.info(f"Валидационный F1-score: {val_f1:.4f}")
    
    # Оценка на тестовой выборке
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    logger.info(f"Тестовая accuracy: {test_accuracy:.4f}")
    logger.info(f"Тестовый F1-score: {test_f1:.4f}")
    
    # Сохранение результатов
    results = {
        'val_accuracy': float(val_accuracy),
        'val_f1': float(val_f1),
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'classification_report': classification_report(y_test, test_pred),
        'feature_importance': dict(zip(vectorizer.get_feature_names_out(), 
                                     np.abs(model.coef_).mean(axis=0)))
    }
    
    # Сохранение результатов
    with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Визуализация confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Сохранение модели и векторизатора
    import joblib
    joblib.dump(model, os.path.join(RESULTS_DIR, 'baseline_model.joblib'))
    joblib.dump(vectorizer, os.path.join(RESULTS_DIR, 'tfidf_vectorizer.joblib'))
    
    return results

def main():
    """Основная функция."""
    # Загрузка данных
    dataset = load_data()
    
    # Обучение модели
    results = train_baseline(
        dataset['train'],
        dataset['validation'],
        dataset['test']
    )
    
    logger.info("Обучение бейзлайна завершено")
    logger.info(f"Результаты сохранены в {RESULTS_DIR}")

if __name__ == "__main__":
    main() 