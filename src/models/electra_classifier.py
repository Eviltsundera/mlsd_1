#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для обучения ELECTRA-base для классификации новостных статей.
"""

import os
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    'model_name': 'google/electra-base-discriminator',
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 1,
    'gradient_accumulation_steps': 4,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'max_length': 512,
    'data_dir': 'data/processed/merged/merged_v1',
    'results_dir': 'models/electra'
}

# Создание директории для результатов
os.makedirs(CONFIG['results_dir'], exist_ok=True)

def load_data():
    """Загрузка и подготовка данных."""
    logger.info("Загрузка данных...")
    dataset = load_from_disk(CONFIG['data_dir'])
    
    # Разделение на train и validation
    train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    
    logger.info(f"Размер обучающей выборки: {len(train_data)}")
    logger.info(f"Размер валидационной выборки: {len(val_data)}")
    logger.info(f"Размер тестовой выборки: {len(dataset['test'])}")
    
    return train_data, val_data, dataset['test']

def create_dataloaders(train_data, val_data, test_data, tokenizer):
    """Создание DataLoader'ов для обучения, валидации и тестирования."""
    
    def tokenize_function(examples):
        """Токенизация текстов и конвертация меток."""
        # Получаем тексты и метки
        texts = [str(text) for text in examples['text']]  # Явное приведение к строке
        labels = examples['harmonized_label']
        
        # Токенизация
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_length'],
            return_tensors=None  # Изменено с 'pt' на None
        )
        
        # Конвертация меток в числовой формат
        label_to_id = {label: idx for idx, label in enumerate(sorted(set(train_data['harmonized_label'])))}
        numeric_labels = [label_to_id[label] for label in labels]
        
        # Преобразование в тензоры
        return {
            'input_ids': torch.tensor(encoded['input_ids']),
            'attention_mask': torch.tensor(encoded['attention_mask']),
            'labels': torch.tensor(numeric_labels)
        }
    
    # Токенизация данных
    train_encoded = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=train_data.column_names
    )
    val_encoded = val_data.map(
        tokenize_function,
        batched=True,
        remove_columns=val_data.column_names
    )
    test_encoded = test_data.map(
        tokenize_function,
        batched=True,
        remove_columns=test_data.column_names
    )
    
    # Настройка формата данных для DataLoader
    train_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Создание DataLoader'ов
    train_dataloader = DataLoader(
        train_encoded,
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_encoded,
        batch_size=CONFIG['batch_size']
    )
    test_dataloader = DataLoader(
        test_encoded,
        batch_size=CONFIG['batch_size']
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def train_epoch(model, train_dataloader, optimizer, scheduler, device):
    """Обучение модели на одной эпохе."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Обучение")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / CONFIG['gradient_accumulation_steps']
        loss.backward()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Логирование в wandb
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })
    
    return total_loss / len(train_dataloader)

def evaluate(model, dataloader, device):
    """Оценка модели на датасете."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Оценка"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1

def main():
    """Основная функция для обучения модели."""
    # Инициализация wandb
    wandb.init(
        project="news-classification",
        config=CONFIG,
        name="electra_classifier"
    )
    
    # Загрузка данных
    train_data, val_data, test_data = load_data()
    
    # Инициализация токенизатора и модели
    tokenizer = ElectraTokenizer.from_pretrained(CONFIG['model_name'])
    model = ElectraForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(set(train_data['harmonized_label']))
    )
    
    # Создание DataLoader'ов
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_data, val_data, test_data, tokenizer
    )
    
    # Настройка устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Настройка оптимизатора и планировщика
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    num_training_steps = len(train_dataloader) * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Обучение модели
    best_val_f1 = 0
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"Эпоха {epoch + 1}/{CONFIG['num_epochs']}")
        
        # Обучение
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        
        # Валидация
        val_accuracy, val_f1 = evaluate(model, val_dataloader, device)
        
        # Логирование метрик
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        })
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation F1: {val_f1:.4f}")
        
        # Сохранение лучшей модели
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_save_path = os.path.join(CONFIG['results_dir'], 'best_model')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logger.info(f"Сохранена лучшая модель с F1-score: {val_f1:.4f}")
    
    # Оценка на тестовом наборе
    test_accuracy, test_f1 = evaluate(model, test_dataloader, device)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    
    # Логирование финальных метрик
    wandb.log({
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    })
    
    # Завершение wandb
    wandb.finish()

if __name__ == "__main__":
    main() 