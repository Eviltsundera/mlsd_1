#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Обучение DistilBERT для классификации новостных статей.
"""

import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW
)
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, classification_report
import wandb
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 4,
    'seed': 42
}

# Пути
MERGED_DATA_DIR = "data/processed/merged/merged"
RESULTS_DIR = "models/bert"
os.makedirs(RESULTS_DIR, exist_ok=True)

def setup_wandb():
    """Инициализация wandb."""
    wandb.init(
        project="news-classification",
        config=CONFIG,
        name="distilbert-baseline"
    )

def load_data():
    """Загрузка датасета."""
    logger.info("Загрузка датасета...")
    dataset = load_from_disk(MERGED_DATA_DIR)
    return dataset

def setup_device():
    """Настройка устройства для обучения."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")
    return device

def tokenize_function(examples, tokenizer):
    """Токенизация текстов."""
    # Преобразуем harmonized_label в числовые метки
    label_to_id = {
        'technology': 0,
        'world': 1,
        'sports': 2,
        'business': 3,
        'entertainment': 4,
        'other': 5
    }
    
    labels = [label_to_id[label] for label in examples['harmonized_label']]
    
    # Токенизируем тексты
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=CONFIG['max_length']
    )
    
    # Добавляем метки к токенизированным данным
    tokenized['labels'] = labels
    return tokenized

def create_dataloaders(dataset, tokenizer):
    """Создание DataLoader'ов."""
    # Словарь для преобразования меток
    label_to_id = {
        'technology': 0,
        'world': 1,
        'sports': 2,
        'business': 3,
        'entertainment': 4,
        'other': 5
    }
    
    def encode_batch(examples):
        """Кодирование батча данных."""
        # Получаем тексты и метки
        texts = [str(example['text']) for example in examples]  # Явное приведение к строке
        labels = [label_to_id[example['harmonized_label']] for example in examples]
        
        # Токенизируем каждый текст отдельно
        encodings = []
        for text in texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        
        # Собираем батч
        batch = {
            'input_ids': torch.stack([enc['input_ids'] for enc in encodings]),
            'attention_mask': torch.stack([enc['attention_mask'] for enc in encodings]),
            'labels': torch.tensor(labels)
        }
        
        return batch
    
    # Создание DataLoader'ов
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=encode_batch
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=encode_batch
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=encode_batch
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def train_epoch(model, train_dataloader, optimizer, scheduler, device):
    """Обучение за одну эпоху."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        # Перемещение данных на устройство
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / CONFIG['gradient_accumulation_steps']
        loss.backward()
        
        total_loss += loss.item()
        
        # Оптимизация
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
    """Оценка модели."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
    
    return accuracy, f1, all_preds, all_labels

def main():
    """Основная функция."""
    # Инициализация wandb
    setup_wandb()
    
    # Загрузка данных
    dataset = load_data()
    
    # Настройка устройства
    device = setup_device()
    
    # Инициализация токенизатора и модели
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=6  # Количество классов: technology, world, sports, business, entertainment, other
    ).to(device)
    
    # Создание DataLoader'ов
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset, tokenizer
    )
    
    # Настройка оптимизатора и планировщика
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    num_training_steps = len(train_dataloader) * CONFIG['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Обучение
    best_val_f1 = 0
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"Эпоха {epoch + 1}/{CONFIG['num_epochs']}")
        
        # Обучение
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        
        # Оценка на валидационной выборке
        val_accuracy, val_f1, _, _ = evaluate(model, val_dataloader, device)
        
        # Логирование метрик
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        })
        
        logger.info(f"Валидационная accuracy: {val_accuracy:.4f}")
        logger.info(f"Валидационный F1-score: {val_f1:.4f}")
        
        # Сохранение лучшей модели
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(os.path.join(RESULTS_DIR, 'best_model'))
            tokenizer.save_pretrained(os.path.join(RESULTS_DIR, 'best_model'))
    
    # Оценка на тестовой выборке
    test_accuracy, test_f1, test_preds, test_labels = evaluate(
        model, test_dataloader, device
    )
    
    # Сохранение результатов
    results = {
        'test_accuracy': float(test_accuracy),
        'test_f1': float(test_f1),
        'classification_report': classification_report(test_labels, test_preds)
    }
    
    with open(os.path.join(RESULTS_DIR, 'bert_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Завершение wandb
    wandb.finish()
    
    logger.info(f"Тестовая accuracy: {test_accuracy:.4f}")
    logger.info(f"Тестовый F1-score: {test_f1:.4f}")

if __name__ == "__main__":
    main() 