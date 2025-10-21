# Corrosion Analysis Project

Проект для анализа коррозии трубопроводного оборудования на основе данных измерений.

## Описание

Этот проект предназначен для анализа коррозионных процессов в трубопроводном оборудовании с использованием методов машинного обучения и статистического анализа.

## Структура проекта

```
corrosion_analysis_project/
├── config/                 # Конфигурационные файлы
│   ├── __init__.py
│   └── database_config.py  # Настройки подключения к БД
├── data/                   # Данные
│   └── corrosion_analysis_cleaned.csv
├── notebooks/              # Jupyter notebooks для анализа
│   ├── 01_data_exploration.ipynb
│   └── 02_correlation_analysis.ipynb
├── src/                    # Исходный код
│   ├── analysis.py         # Основные функции анализа
│   └── database.py         # Работа с базой данных
├── requirements.txt        # Зависимости Python
└── README.md              # Этот файл
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd corrosion_analysis_project
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
```

3. Активируйте виртуальное окружение:
```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

4. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Анализ данных

Запустите Jupyter notebooks для исследования данных:

```bash
jupyter notebook notebooks/
```

### Работа с базой данных

```python
from src.database import load_corrosion_data, load_raw_data

# Загрузка обработанных данных
df = load_corrosion_data()

# Загрузка сырых данных
raw_df = load_raw_data()
```

## Зависимости

- pandas - работа с данными
- numpy - численные вычисления
- matplotlib, seaborn - визуализация
- SQLAlchemy - работа с базой данных
- psycopg2 - подключение к PostgreSQL

## Лицензия

Этот проект предназначен для внутреннего использования.
