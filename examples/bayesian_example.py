"""
Пример использования байесовских методов для анализа коррозии KK-2

Этот скрипт демонстрирует:
1. Bayesian Ridge Regression с оценкой неопределенности
2. Байесовскую оптимизацию гиперпараметров Random Forest
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from database import load_corrosion_data_new as load_data
from analysis import AdvancedCorrosionAnalyzer
from bayesian_analysis import BayesianCorrosionAnalyzer

# Настройки
TARGET = 'worst_corrosion_rate_mm_per_year'
INSTALLATION_FILTER = 'KK-2'

# Загрузка данных
print("Загрузка данных...")
DF = load_data()
DF = DF[DF['installation'] == INSTALLATION_FILTER]
print(f"Данные для {INSTALLATION_FILTER}: {len(DF):,} строк")

# Подготовка данных с инженерными признаками
print("\nПодготовка данных...")
analyzer = AdvancedCorrosionAnalyzer(DF)
added_cols = analyzer.add_engineered_indices()
DF_prepared = analyzer.data.copy()

# Инициализация байесовского анализатора
print("\nИнициализация байесовского анализатора...")
bayesian_analyzer = BayesianCorrosionAnalyzer(DF_prepared)
bayesian_analyzer.set_target(TARGET)

# Выбор признаков
feature_columns = [
    'h2s_content', 
    'h2s_water_ratio',
    'h2s_aggressiveness_index', 
    'material_resistance_score', 
    'wall_thickness', 
    'equipment_age_years',
    'component_type_id',
    'corrosion_protection_index',
    'stress_corrosion_index'
]

print(f"Используем {len(feature_columns)} признаков")

# 1. Bayesian Ridge Regression
print("\n" + "="*70)
print("1️⃣ BAYESIAN RIDGE REGRESSION")
print("="*70)
br_result = bayesian_analyzer.train_bayesian_ridge(
    feature_columns=feature_columns,
    test_size=0.2,
    random_state=42,
    n_iter=300
)

# Визуализация неопределенности
try:
    bayesian_analyzer.plot_uncertainty('bayesian_ridge')
except Exception as e:
    print(f"⚠️ Не удалось визуализировать неопределенность: {e}")

# 2. Байесовская оптимизация Random Forest
print("\n" + "="*70)
print("2️⃣ БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ RANDOM FOREST")
print("="*70)
print("⚠️ Это может занять несколько минут...")
rf_optimized = bayesian_analyzer.optimize_random_forest_bayesian(
    feature_columns=feature_columns,
    n_trials=30,  # Можно увеличить до 50-100 для лучших результатов
    test_size=0.2,
    random_state=42,
    cv_folds=5,
    use_optuna=True
)

# 3. Сравнение всех методов
print("\n" + "="*70)
print("3️⃣ СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
print("="*70)
comparison = bayesian_analyzer.compare_bayesian_methods(
    feature_columns=feature_columns,
    test_size=0.2,
    random_state=42,
    n_trials=30
)

print("\n✅ Анализ завершен!")
print(f"\nРезультаты сохранены в bayesian_analyzer.results")
print(f"Доступные ключи: {list(bayesian_analyzer.results.keys())}")

