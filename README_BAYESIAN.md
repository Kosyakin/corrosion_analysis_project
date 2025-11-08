# Байесовские методы для анализа коррозии

## Описание

Модуль `bayesian_analysis.py` предоставляет байесовские методы для анализа коррозии:

1. **Bayesian Ridge Regression** - линейная регрессия с байесовским подходом и оценкой неопределенности предсказаний
2. **Байесовская оптимизация гиперпараметров** - оптимизация Random Forest с помощью Optuna или scikit-optimize

## Установка

```bash
pip install -r requirements.txt
```

Требуемые библиотеки:
- `scikit-learn` (уже установлено)
- `optuna` (рекомендуется) или `scikit-optimize`

## Быстрый старт

### Пример 1: Bayesian Ridge Regression

```python
from src.bayesian_analysis import BayesianCorrosionAnalyzer
from src.analysis import AdvancedCorrosionAnalyzer
from src.database import load_corrosion_data_new as load_data

# Загрузка данных
DF = load_data()
DF = DF[DF['installation'] == 'KK-2']

# Подготовка данных
analyzer = AdvancedCorrosionAnalyzer(DF)
analyzer.add_engineered_indices()
DF_prepared = analyzer.data.copy()

# Байесовский анализ
bayesian_analyzer = BayesianCorrosionAnalyzer(DF_prepared)
bayesian_analyzer.set_target('worst_corrosion_rate_mm_per_year')

# Обучение Bayesian Ridge
feature_columns = [
    'h2s_content', 'h2s_water_ratio', 'h2s_aggressiveness_index',
    'material_resistance_score', 'wall_thickness', 'equipment_age_years',
    'component_type_id', 'corrosion_protection_index', 'stress_corrosion_index'
]

br_result = bayesian_analyzer.train_bayesian_ridge(
    feature_columns=feature_columns,
    test_size=0.2,
    random_state=42
)

# Визуализация неопределенности
bayesian_analyzer.plot_uncertainty('bayesian_ridge')
```

### Пример 2: Байесовская оптимизация Random Forest

```python
# Оптимизация гиперпараметров RF
rf_optimized = bayesian_analyzer.optimize_random_forest_bayesian(
    feature_columns=feature_columns,
    n_trials=50,  # Количество попыток оптимизации
    test_size=0.2,
    random_state=42,
    cv_folds=5,
    use_optuna=True  # Использовать Optuna (или False для scikit-optimize)
)

print(f"Лучшие параметры: {rf_optimized['best_params']}")
print(f"R²: {rf_optimized['metrics']['r2']:.4f}")
```

### Пример 3: Сравнение всех методов

```python
# Сравнение Bayesian Ridge и оптимизированного RF
comparison = bayesian_analyzer.compare_bayesian_methods(
    feature_columns=feature_columns,
    test_size=0.2,
    random_state=42,
    n_trials=30
)
```

## Основные методы

### `BayesianCorrosionAnalyzer`

#### `train_bayesian_ridge(feature_columns, test_size=0.2, random_state=42, n_iter=300)`

Обучение Bayesian Ridge регрессии.

**Возвращает:**
- `model`: обученная модель
- `metrics`: метрики (r2, mae, rmse)
- `uncertainty`: стандартные отклонения предсказаний
- `y_pred`: предсказания на тесте
- `y_test`: истинные значения на тесте

#### `optimize_random_forest_bayesian(feature_columns, n_trials=50, test_size=0.2, random_state=42, cv_folds=5, use_optuna=True)`

Байесовская оптимизация гиперпараметров Random Forest.

**Возвращает:**
- `best_params`: лучшие гиперпараметры
- `best_score`: лучшее значение метрики (CV)
- `best_model`: обученная модель с лучшими параметрами
- `metrics`: метрики на тестовой выборке

#### `compare_bayesian_methods(feature_columns, test_size=0.2, random_state=42, n_trials=30)`

Сравнение Bayesian Ridge и оптимизированного Random Forest с базовым RF.

**Возвращает:** DataFrame с результатами сравнения

#### `plot_uncertainty(result_key='bayesian_ridge')`

Визуализация неопределенности предсказаний Bayesian Ridge.

## Преимущества байесовских методов

1. **Оценка неопределенности** - можно получить не только предсказание, но и его надежность
2. **Автоматическая регуляризация** - Bayesian Ridge автоматически подбирает параметры регуляризации
3. **Эффективная оптимизация** - байесовская оптимизация находит лучшие гиперпараметры быстрее, чем GridSearch

## Результаты

Результаты сохраняются в `bayesian_analyzer.results`:

- `results['bayesian_ridge']` - результаты Bayesian Ridge
- `results['optimized_rf']` - результаты оптимизированного RF
- `results['comparison']` - сравнение методов

## Примечания

- Bayesian Ridge может быть медленнее обычного Ridge, но дает оценку неопределенности
- Байесовская оптимизация может занять несколько минут (зависит от `n_trials`)
- Для больших датасетов рекомендуется использовать Optuna (быстрее, чем scikit-optimize)

## Примеры использования

См. `examples/bayesian_example.py` для полного примера использования.

