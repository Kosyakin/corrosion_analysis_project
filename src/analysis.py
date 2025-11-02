import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Предполагаем, что TARGET определена как 'corrosion_rate_mm_per_year'
TARGET = 'corrosion_rate_mm_per_year'


class CorrosionAnalyzer:
    """Класс для анализа коррозии с двумя основными методами."""
    
    def __init__(self, df: pd.DataFrame, target: str = TARGET):
        if target not in df.columns:
            raise ValueError(f"Целевая переменная '{target}' отсутствует в данных")
        self.df = df.copy()
        self.target = target
    
    def compute_target_correlations(self, feature_cols: list | None = None,
                                  top_k: int = 30, method: str = 'spearman',
                                  return_best_features: bool = True) -> tuple[pd.DataFrame, list] | pd.DataFrame:
        """
        Вычисляет корреляции признаков с целевой переменной и возвращает лучшие признаки
        
        Parameters:
        -----------
        feature_cols : list, optional
            Список признаков для анализа. Если None, используются все числовые колонки
        top_k : int
            Количество топ-признаков для возврата
        method : str
            Метод корреляции ('pearson' или 'spearman')
        return_best_features : bool
            Если True, возвращает кортеж (DataFrame, list), иначе только DataFrame
        
        Returns:
        --------
        tuple[pd.DataFrame, list] or pd.DataFrame
            DataFrame с корреляциями и список лучших признаков, либо только DataFrame
        """
        # Только числовые признаки
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        
        if feature_cols is not None and len(feature_cols) > 0:
            feature_cols = [c for c in feature_cols if c in num_df.columns and c != self.target]
        else:
            feature_cols = [c for c in num_df.columns if c != self.target]

        # Очистка по цели
        valid = num_df[self.target].notna()
        num_df = num_df.loc[valid]

        # Стандартное ограничение цели
        y = num_df[self.target]

        # Корреляции
        if method == 'pearson':
            corr_series = num_df[feature_cols].corrwith(y)
        elif method == 'spearman':
            corr_series = num_df[feature_cols].rank().corrwith(y.rank())
        else:
            raise ValueError("method должен быть 'pearson' или 'spearman'")

        # Сортировка по абсолютному значению
        res = corr_series.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        if top_k:
            res = res.head(top_k)

        # Создание DataFrame с результатами
        out_df = pd.DataFrame({'feature': res.index, 'corr': res.values})
        
        # Список лучших признаков
        best_features = res.index.tolist()
        
        # Вывод результатов
        print("=" * 70)
        print(f"ТОП-{top_k} ПРИЗНАКОВ ПО КОРРЕЛЯЦИИ С {self.target}:")
        print("=" * 70)
        for i, (feature, corr) in enumerate(zip(out_df['feature'], out_df['corr']), 1):
            significance = "***" if abs(corr) > 0.3 else "** " if abs(corr) > 0.2 else "*  " if abs(corr) > 0.1 else "   "
            direction = "🡅" if corr > 0 else "🡇"
            print(f"{i:2d}. {significance} {feature:30} : {corr:+.4f} {direction}")
        
        print(f"\n📊 Статистика:")
        print(f"   Всего проанализировано признаков: {len(feature_cols)}")
        print(f"   Возвращено топ-признаков: {len(best_features)}")
        print(f"   Максимальная корреляция: {out_df['corr'].abs().max():.4f}")
        print(f"   Минимальная корреляция: {out_df['corr'].abs().min():.4f}")
        
        if return_best_features:
            return out_df, best_features
        else:
            return out_df
    
    def run_experiment(self, columns: list, experiment_name: str,
                      test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """
        Запускает эксперимент машинного обучения на предварительно отфильтрованных данных
        
        Parameters:
        -----------
        columns : list
            Список колонок для использования в качестве признаков
        experiment_name : str
            Название эксперимента для идентификации
        test_size : float
            Доля тестовой выборки (по умолчанию 0.2)
        random_state : int
            Seed для воспроизводимости (по умолчанию 42)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame с результатами эксперимента
        """
        if not isinstance(columns, (list, tuple)) or len(columns) == 0:
            raise ValueError('columns должен быть непустым списком колонок')
        
        # Проверяем существование колонок
        valid_columns = [col for col in columns if col in self.df.columns]
        if len(valid_columns) != len(columns):
            missing = set(columns) - set(valid_columns)
            print(f"⚠️  Предупреждение: отсутствуют колонки: {missing}")
        
        # Фильтруем данные
        data = self.df[valid_columns + [self.target]].dropna()
        
        if len(data) < 10:
            raise ValueError(f"Слишком мало данных для обучения: {len(data)} строк")
        
        context = f"n_samples={len(data)}"
        
        # Подготовка данных
        X = data[valid_columns]
        y = data[self.target]
        
        if len(X) < 50:
            print(f"⚠️  Предупреждение: мало данных для обучения: {len(X)} строк")
        
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Ridge
        ridge = Pipeline([
            ('scaler', StandardScaler(with_mean=False)), 
            ('model', Ridge(alpha=1.0, random_state=random_state))
        ])
        ridge.fit(X_tr, y_tr)
        pr = ridge.predict(X_va)

        # RandomForest
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
        rf.fit(X_tr, y_tr)
        prf = rf.predict(X_va)

        # Собираем результаты
        res = pd.DataFrame([
            {
                'experiment': experiment_name, 
                'context': context, 
                'model': 'Ridge',
                'MAE': mean_absolute_error(y_va, pr), 
                'RMSE': np.sqrt(mean_squared_error(y_va, pr)),
                'R2': r2_score(y_va, pr), 
                'n_samples': len(X),
                'n_features': len(valid_columns)
            },
            {
                'experiment': experiment_name, 
                'context': context, 
                'model': 'RandomForest',
                'MAE': mean_absolute_error(y_va, prf), 
                'RMSE': np.sqrt(mean_squared_error(y_va, prf)),
                'R2': r2_score(y_va, prf), 
                'n_samples': len(X),
                'n_features': len(valid_columns)
            },
        ])
        
        # Вывод результатов
        print("=" * 60)
        print(f"📊 РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: {experiment_name}")
        print("=" * 60)
        print(f"   Данные: {len(data)} наблюдений, {len(valid_columns)} признаков")
        print(f"   Признаки: {valid_columns}")
        print("\n   Метрики качества:")
        for _, row in res.iterrows():
            print(f"     {row['model']:12} | R² = {row['R2']:7.4f} | MAE = {row['MAE']:7.4f} | RMSE = {row['RMSE']:7.4f}")
        
        return res

# Вспомогательная функция для построения матрицы корреляций
def plot_correlation_matrix(df: pd.DataFrame, features: list, target: str = TARGET, 
                          figsize: tuple = (12, 10)):
    """Визуализация матрицы корреляций для топ-признаков"""
    if len(features) < 2:
        print("⚠️  Для матрицы корреляций нужно минимум 2 признака")
        return
    
    # Берем только существующие признаки
    valid_features = [f for f in features if f in df.columns]
    corr_data = df[valid_features + [target]].dropna()
    
    if len(corr_data) < 10:
        print("⚠️  Недостаточно данных для построения матрицы корреляций")
        return
    
    plt.figure(figsize=figsize)
    corr_matrix = corr_data.corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
               square=True, fmt='.2f', cbar_kws={'shrink': .8})
    plt.title(f'Матрица корреляций Спирмена\n({len(corr_data)} наблюдений)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()