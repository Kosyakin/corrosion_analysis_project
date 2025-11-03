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


class AdvancedCorrosionAnalyzer:
    """
    Универсальный класс для анализа коррозии с гибкой настройкой параметров
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame с данными для анализа
        """
        self.data = data.copy()
        self.results = {}
    
    def set_target(self, target_column: str):
        """
        Установка целевой переменной
        
        Parameters:
        -----------
        target_column : str
            Название колонки с целевой переменной
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Колонка '{target_column}' отсутствует в данных")
        self.target = target_column
        return self
    
    def analyze_correlations(self, feature_columns: list, 
                           method: str = 'spearman',
                           top_k: int = 20,
                           plot_matrix: bool = True,
                           figsize: tuple = (12, 10)) -> pd.DataFrame:
        """
        Анализ корреляций между признаками и целевой переменной
        
        Parameters:
        -----------
        feature_columns : list
            Список колонок для анализа корреляций
        method : str
            Метод корреляции ('pearson', 'spearman', 'kendall')
        top_k : int
            Количество топ-признаков для возврата
        plot_matrix : bool
            Строить ли матрицу корреляций
        figsize : tuple
            Размер графика матрицы корреляций
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с результатами корреляций
        """
        if not hasattr(self, 'target'):
            raise ValueError("Сначала установите целевую переменную с помощью set_target()")
        
        # Фильтруем только существующие числовые колонки
        numeric_data = self.data.select_dtypes(include=[np.number])
        valid_features = [f for f in feature_columns if f in numeric_data.columns and f != self.target]
        
        if not valid_features:
            raise ValueError("Нет валидных числовых признаков для анализа")
        
        # Очистка данных
        analysis_data = self.data[valid_features + [self.target]].dropna()
        
        if len(analysis_data) < 10:
            raise ValueError(f"Недостаточно данных после очистки: {len(analysis_data)} строк")
        
        # Вычисление корреляций
        corr_results = []
        for feature in valid_features:
            try:
                if method == 'pearson':
                    corr = analysis_data[feature].corr(analysis_data[self.target])
                    p_value = self._calculate_p_value(analysis_data[feature], analysis_data[self.target])
                elif method == 'spearman':
                    corr, p_value = spearmanr(analysis_data[feature], analysis_data[self.target])
                else:
                    raise ValueError("Метод должен быть 'pearson' или 'spearman'")
                
                if not np.isnan(corr):
                    corr_results.append({
                        'feature': feature,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'n_observations': len(analysis_data)
                    })
            except Exception as e:
                print(f"⚠️ Ошибка для {feature}: {e}")
        
        if not corr_results:
            print("❌ Не удалось вычислить корреляции")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(corr_results).sort_values('abs_correlation', ascending=False).head(top_k)
        
        # Визуализация матрицы корреляций
        if plot_matrix and len(result_df) > 1:
            self._plot_correlation_matrix(analysis_data, result_df, method, figsize)
        
        # Вывод результатов
        self._print_correlation_results(result_df, method)
        
        self.results['correlations'] = result_df
        return result_df
    
    def evaluate_models(self, feature_columns: list,
                       model_types: list = ['ridge', 'random_forest'],
                       test_size: float = 0.2,
                       random_state: int = 42,
                       rf_estimators: int = 100,
                       return_importances: bool = False) -> pd.DataFrame:
        """
        Оценка моделей машинного обучения
        
        Parameters:
        -----------
        feature_columns : list
            Список колонок для использования в качестве признаков
        model_types : list
            Список моделей для оценки ['ridge', 'random_forest']
        test_size : float
            Доля тестовой выборки
        random_state : int
            Seed для воспроизводимости
        rf_estimators : int
            Количество деревьев в Random Forest
        return_importances : bool
            Возвращать ли важности признаков
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с результатами моделей
        """
        if not hasattr(self, 'target'):
            raise ValueError("Сначала установите целевую переменную с помощью set_target()")
        
        # Проверка и подготовка данных
        valid_features = [f for f in feature_columns if f in self.data.columns]
        if len(valid_features) != len(feature_columns):
            missing = set(feature_columns) - set(valid_features)
            print(f"⚠️ Отсутствуют колонки: {missing}")
        
        model_data = self.data[valid_features + [self.target]].dropna()
        
        if len(model_data) < 50:
            raise ValueError(f"Слишком мало данных для обучения: {len(model_data)} строк")
        
        # Подготовка данных
        X = model_data[valid_features]
        y = model_data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = []
        importances = {}
        
        # Обучение моделей
        for model_type in model_types:
            try:
                if model_type == 'ridge':
                    model, metrics = self._train_ridge(X_train, X_test, y_train, y_test, random_state)
                elif model_type == 'random_forest':
                    model, metrics, feature_importance = self._train_random_forest(
                        X_train, X_test, y_train, y_test, valid_features, rf_estimators, random_state
                    )
                    if return_importances:
                        importances[model_type] = feature_importance
                else:
                    print(f"⚠️ Неизвестный тип модели: {model_type}")
                    continue
                
                results.append({
                    'model': model_type,
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'n_features': len(valid_features),
                    'n_samples': len(model_data),
                    'features': ', '.join(valid_features[:3]) + ('...' if len(valid_features) > 3 else '')
                })
                
            except Exception as e:
                print(f"❌ Ошибка в модели {model_type}: {e}")
        
        results_df = pd.DataFrame(results)
        
        # Визуализация результатов
        self._plot_model_comparison(results_df)
        
        # Вывод результатов
        self._print_model_results(results_df)
        
        self.results['models'] = results_df
        if return_importances:
            self.results['importances'] = importances
            return results_df, importances
        
        return results_df
    
    def compare_feature_sets(self, feature_sets: dict,
                           test_size: float = 0.2,
                           random_state: int = 42) -> pd.DataFrame:
        """
        Сравнение разных наборов признаков
        
        Parameters:
        -----------
        feature_sets : dict
            Словарь вида {'набор_признаков': [список_признаков]}
        test_size : float
            Доля тестовой выборки
        random_state : int
            Seed для воспроизводимости
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с результатами сравнения
        """
        all_results = []
        
        for set_name, features in feature_sets.items():
            print(f"\n🔬 Тестируем набор: {set_name}")
            print(f"   Признаки: {features}")
            
            try:
                results = self.evaluate_models(
                    feature_columns=features,
                    test_size=test_size,
                    random_state=random_state
                )
                
                # Добавляем имя набора к результатам
                for result in results.to_dict('records'):
                    result['feature_set'] = set_name
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ Ошибка для набора {set_name}: {e}")
        
        if not all_results:
            print("❌ Не удалось выполнить ни одного эксперимента")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(all_results)
        
        # Визуализация сравнения
        self._plot_feature_set_comparison(comparison_df)
        
        return comparison_df

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    def _calculate_p_value(self, x, y):
        """Расчет p-value для корреляции Пирсона"""
        from scipy.stats import pearsonr
        try:
            _, p_value = pearsonr(x, y)
            return p_value
        except:
            return 1.0
    
    def _plot_correlation_matrix(self, data, corr_df, method, figsize):
        """Визуализация матрицы корреляций"""
        top_features = corr_df.head(10)['feature'].tolist()
        
        if len(top_features) < 2:
            return
            
        plt.figure(figsize=figsize)
        corr_matrix = data[top_features + [self.target]].corr(method=method)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': .8})
        plt.title(f'Матрица корреляций {method.upper()}\n({len(data)} наблюдений)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    def _print_correlation_results(self, result_df, method):
        """Вывод результатов корреляционного анализа"""
        print("=" * 70)
        print(f"ТОП-{len(result_df)} ПРИЗНАКОВ ПО КОРРЕЛЯЦИИ {method.upper()}")
        print("=" * 70)
        
        significant = result_df[result_df['significant']]
        non_significant = result_df[~result_df['significant']]
        
        if len(significant) > 0:
            print("\n📈 СТАТИСТИЧЕСКИ ЗНАЧИМЫЕ (p < 0.05):")
            for _, row in significant.iterrows():
                direction = "🡅" if row['correlation'] > 0 else "🡇"
                stars = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
                print(f"  {row['feature']:30} | r = {row['correlation']:7.3f} {stars} {direction}")
        
        print(f"\n📊 Статистика:")
        print(f"   Всего значимых корреляций: {len(significant)}")
        print(f"   Максимальная корреляция: {result_df['correlation'].abs().max():.3f}")
        print(f"   Минимальная корреляция: {result_df['correlation'].abs().min():.3f}")
    
    def _train_ridge(self, X_train, X_test, y_train, y_test, random_state):
        """Обучение Ridge регрессии"""
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', Ridge(alpha=1.0, random_state=random_state))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return model, metrics
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test, features, n_estimators, random_state):
        """Обучение Random Forest"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Важности признаков
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, metrics, importance_df
    
    def _plot_model_comparison(self, results_df):
        """Визуализация сравнения моделей"""
        if len(results_df) < 2:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['r2', 'mae', 'rmse']
        titles = ['R² (больше → лучше)', 'MAE (меньше → лучше)', 'RMSE (меньше → лучше)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            axes[idx].bar(results_df['model'], results_df[metric], 
                         color=['lightblue', 'lightcoral', 'lightgreen'][:len(results_df)])
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
            
            for i, v in enumerate(results_df[metric]):
                axes[idx].text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Сравнение моделей машинного обучения', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _print_model_results(self, results_df):
        """Вывод результатов моделей"""
        print("=" * 60)
        print("📊 РЕЗУЛЬТАТЫ МОДЕЛЕЙ")
        print("=" * 60)
        
        for _, row in results_df.iterrows():
            print(f"   {row['model']:15} | R² = {row['r2']:7.4f} | "
                  f"MAE = {row['mae']:7.4f} | RMSE = {row['rmse']:7.4f}")
        
        best_model = results_df.loc[results_df['r2'].idxmax()]
        print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ: {best_model['model']} (R² = {best_model['r2']:.4f})")
    
    def _plot_feature_set_comparison(self, comparison_df):
        """Визуализация сравнения наборов признаков"""
        if len(comparison_df) < 2:
            return
            
        pivot_df = comparison_df.pivot(index='feature_set', columns='model', values='r2')
        
        plt.figure(figsize=(12, 6))
        pivot_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Сравнение наборов признаков по качеству R²', fontweight='bold')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()