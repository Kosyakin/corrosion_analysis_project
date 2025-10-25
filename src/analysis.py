import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any

class CorrosionAnalyzer:
    """Класс для анализа данных коррозии трубопроводов"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Инициализация анализатора с данными
        
        Args:
            data: DataFrame с данными коррозии
        """
        self.data = data.copy()
        self.original_data = data.copy()
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Получение базовой информации о данных"""
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Получение сводной статистики по числовым колонкам"""
        return self.data.describe()
    
    def plot_corrosion_distribution(self, column: str = 'corrosion_rate', 
                                  figsize: tuple = (12, 6)) -> None:
        """Построение распределения коррозии"""
        if column not in self.data.columns:
            print(f"Колонка '{column}' не найдена в данных")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Гистограмма
        self.data[column].hist(bins=30, ax=axes[0], alpha=0.7)
        axes[0].set_title(f'Распределение {column}')
        axes[0].set_xlabel(column)
        axes[0].set_ylabel('Частота')
        
        # Box plot
        self.data.boxplot(column=column, ax=axes[1])
        axes[1].set_title(f'Box plot для {column}')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_by_equipment(self, equipment_col: str = 'equipment') -> pd.DataFrame:
        """Анализ коррозии по типам оборудования"""
        if equipment_col not in self.data.columns:
            print(f"Колонка '{equipment_col}' не найдена в данных")
            return pd.DataFrame()
            
        return self.data.groupby(equipment_col).agg({
            'corrosion_rate': ['count', 'mean', 'std', 'min', 'max']
        }).round(3)
    
    def filter_data(self, **filters) -> 'CorrosionAnalyzer':
        """Фильтрация данных по заданным условиям"""
        filtered_data = self.data.copy()
        
        for column, value in filters.items():
            if column in filtered_data.columns:
                if isinstance(value, tuple) and len(value) == 2:
                    # Диапазон значений
                    filtered_data = filtered_data[
                        (filtered_data[column] >= value[0]) & 
                        (filtered_data[column] <= value[1])
                    ]
                else:
                    # Точное значение или список значений
                    if isinstance(value, list):
                        filtered_data = filtered_data[filtered_data[column].isin(value)]
                    else:
                        filtered_data = filtered_data[filtered_data[column] == value]
        
        return CorrosionAnalyzer(filtered_data)
    
    def reset_data(self) -> 'CorrosionAnalyzer':
        """Сброс к исходным данным"""
        return CorrosionAnalyzer(self.original_data)
    
    def safe_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """
        Безопасный расчет корреляции с обработкой NaN значений
        
        Args:
            x: Первая переменная
            y: Вторая переменная
            
        Returns:
            Коэффициент корреляции или NaN если расчет невозможен
        """
        # Удаляем NaN значения
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2:
            return np.nan
        
        # Проверяем стандартное отклонение
        if x_clean.std() == 0 or y_clean.std() == 0:
            return np.nan
        
        return np.corrcoef(x_clean, y_clean)[0, 1]
    
    def get_correlations_with_target(self, target_col: str = 'corrosion_rate', 
                                   exclude_zero_variance: bool = True) -> Dict[str, float]:
        """
        Получение корреляций всех параметров с целевой переменной
        
        Args:
            target_col: Название целевой колонки
            exclude_zero_variance: Исключать параметры с нулевой вариацией
            
        Returns:
            Словарь с корреляциями
        """
        if target_col not in self.data.columns:
            raise ValueError(f"Целевая колонка '{target_col}' не найдена в данных")
        
        correlations = {}
        target = self.data[target_col]
        
        for col in self.data.columns:
            if col != target_col and self.data[col].dtype in ['float64', 'int64']:
                if exclude_zero_variance and self.data[col].std() == 0:
                    continue
                
                corr = self.safe_correlation(self.data[col], target)
                if not np.isnan(corr):
                    correlations[col] = corr
        
        return correlations

