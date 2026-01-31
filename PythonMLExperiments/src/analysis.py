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
    
    def add_engineered_indices(self) -> list:
        """
        Добавление инженерных индексов на основе физико-химических соображений
        
        Добавляет следующие признаки:
        - h2s_aqueous_exposure: доступность H2S в водной фазе
        - co2_aqueous_temp_index: CO2 в воде с температурной коррекцией
        - acid_load_aqueous: кислотная нагрузка в воде
        - chloride_aqueous: хлоридная агрессивность в воде
        - oxygen_aqueous: содержание кислорода в воде
        - mixed_acid_gas_index: комбинированный индекс кислотных газов
        - aggressiveness_per_resistance: соотношение агрессивности и стойкости материала
        - protection_gap: разрыв между агрессивностью и защитой
        - pitting_chloride_index: индекс питтинговой коррозии от хлоридов
        - hoop_stress_proxy: прокси-напряжение оболочки
        - material_adjusted_stress: напряжение с учетом материала
        
        Returns:
        --------
        list
            Список добавленных колонок
        """
        eps = 1e-6
        T0 = 25.0
        kT = 0.025  # температурная чувствительность для CO2 (приближенная)
        
        df = self.data
        
        # Проверка наличия необходимых колонок
        required_cols = [
            'h2s_content', 'h2s_water_ratio', 'water_content', 'co2_content',
            'operating_temperature', 'total_acidity_index', 'chloride_aggressiveness',
            'oxygen_content', 'corrosion_aggressiveness_index', 'material_resistance_score',
            'corrosion_protection_index', 'pitting_corrosion_index',
            'diameter_to_thickness_ratio', 'operating_pressure', 'nominal_thickness_mmc',
            'worst_thickness_measurement', 'equipment_age_years', 'stress_corrosion_index'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Предупреждение: отсутствуют колонки {missing_cols}")
            print("   Некоторые инженерные индексы могут быть не созданы")
        
        # Доступность агрессивных компонентов в водной фазе
        if all(col in df.columns for col in ['h2s_content', 'h2s_water_ratio', 'water_content']):
            df['h2s_aqueous_exposure'] = df['h2s_content'] * df['h2s_water_ratio'] * df['water_content']
        
        if all(col in df.columns for col in ['co2_content', 'water_content', 'operating_temperature']):
            df['co2_aqueous_temp_index'] = df['co2_content'] * df['water_content'] * np.exp(kT * (df['operating_temperature'] - T0))
        
        # Кислотная/хлоридная нагрузка в воде
        if all(col in df.columns for col in ['total_acidity_index', 'water_content']):
            df['acid_load_aqueous'] = df['total_acidity_index'] * df['water_content']
        
        if all(col in df.columns for col in ['chloride_aggressiveness', 'water_content']):
            df['chloride_aqueous'] = df['chloride_aggressiveness'] * df['water_content']
        
        if all(col in df.columns for col in ['oxygen_content', 'water_content']):
            df['oxygen_aqueous'] = df['oxygen_content'] * df['water_content']
        
        # Комбинированные индексы среды/материала/защиты
        if all(col in df.columns for col in ['h2s_content', 'co2_content']):
            df['mixed_acid_gas_index'] = df['h2s_content'] + 0.3 * df['co2_content']
        
        if all(col in df.columns for col in ['corrosion_aggressiveness_index', 'material_resistance_score']):
            df['aggressiveness_per_resistance'] = df['corrosion_aggressiveness_index'] / (df['material_resistance_score'] + eps)
        
        if all(col in df.columns for col in ['corrosion_aggressiveness_index', 'corrosion_protection_index']):
            df['protection_gap'] = (df['corrosion_aggressiveness_index'] - df['corrosion_protection_index']).clip(lower=0)
        
        if all(col in df.columns for col in ['pitting_corrosion_index']) and 'chloride_aqueous' in df.columns:
            df['pitting_chloride_index'] = df['pitting_corrosion_index'] * df['chloride_aqueous']
        
        # Прокси-напряжение и деградация
        if all(col in df.columns for col in ['diameter_to_thickness_ratio', 'operating_pressure']):
            df['hoop_stress_proxy'] = df['diameter_to_thickness_ratio'] * df['operating_pressure']
               
        if all(col in df.columns for col in ['stress_corrosion_index', 'material_resistance_score']):
            df['material_adjusted_stress'] = df['stress_corrosion_index'] / (df['material_resistance_score'] + eps)
        
        # Определяем список добавленных колонок
        expected_cols = [
            'h2s_aqueous_exposure', 'co2_aqueous_temp_index', 'acid_load_aqueous', 'chloride_aqueous',
            'oxygen_aqueous', 'mixed_acid_gas_index', 'aggressiveness_per_resistance', 'protection_gap',
            'pitting_chloride_index', 'hoop_stress_proxy',
            'material_adjusted_stress'
        ]
        
        added_cols = [col for col in expected_cols if col in df.columns]
        
        # Обновляем self.data
        self.data = df
        
        if added_cols:
            print(f"✅ Добавлены инженерные индексы: {len(added_cols)} колонок")
            print(f"   Колонки: {added_cols}")
        else:
            print("⚠️ Не удалось добавить инженерные индексы (отсутствуют необходимые колонки)")
        
        return added_cols
    
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
        
        # Визуализация результатов отключена (сравнение Ridge vs Random Forest)
        # self._plot_model_comparison(results_df)
        
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

        # Сводная таблица по наборам признаков для Random Forest
        try:
            rf_df = comparison_df[comparison_df['model'] == 'random_forest'][['feature_set', 'r2', 'mae']].copy()
            if not rf_df.empty:
                summary_df = rf_df.rename(columns={
                    'feature_set': 'Набор параметров',
                    'r2': 'R2 Random Forest',
                    'mae': 'MAE Random Forest'
                })
                print("\n📋 Сводная таблица по наборам признаков (Random Forest):")
                print(summary_df.set_index('Набор параметров').round(4).to_string())
                # Сохраняем в результаты для дальнейшего использования при необходимости
                self.results['feature_sets_summary'] = summary_df
        except Exception as e:
            print(f"⚠️ Не удалось сформировать сводную таблицу: {e}")
        
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