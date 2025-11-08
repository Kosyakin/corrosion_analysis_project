import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта библиотек для байесовской оптимизации
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna не установлена. Для байесовской оптимизации установите: pip install optuna")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    # Не выводим предупреждение здесь, только при попытке использовать


class BayesianCorrosionAnalyzer:
    """
    Класс для байесовского анализа коррозии с оценкой неопределенности предсказаний
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
        self.target = None
    
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
    
    def train_bayesian_ridge(self, feature_columns: list,
                            test_size: float = 0.2,
                            random_state: int = 42,
                            n_iter: int = 300,
                            compute_score: bool = True) -> dict:
        """
        Обучение Bayesian Ridge регрессии с оценкой неопределенности
        
        Parameters:
        -----------
        feature_columns : list
            Список колонок для использования в качестве признаков
        test_size : float
            Доля тестовой выборки
        random_state : int
            Seed для воспроизводимости
        n_iter : int
            Количество итераций для оптимизации
        compute_score : bool
            Вычислять ли score модели
            
        Returns:
        --------
        dict
            Словарь с результатами:
            - model: обученная модель
            - metrics: метрики (r2, mae, rmse)
            - uncertainty: стандартные отклонения предсказаний
            - y_pred: предсказания на тесте
            - y_test: истинные значения на тесте
        """
        if not self.target:
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
        
        # Обучение Bayesian Ridge
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', BayesianRidge(
                n_iter=n_iter,
                compute_score=compute_score,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            ))
        ])
        
        model.fit(X_train, y_train)
        
        # Предсказания с оценкой неопределенности
        y_pred_mean = model.predict(X_test)
        
        # Получение стандартного отклонения (если возможно)
        # BayesianRidge не предоставляет напрямую std для предсказаний,
        # но можем оценить неопределенность через дисперсию параметров
        try:
            # Получаем модель из pipeline
            br_model = model.named_steps['model']
            # Оценка неопределенности через дисперсию предсказаний
            # Используем дисперсию параметров модели
            X_test_scaled = model.named_steps['scaler'].transform(X_test)
            # Оценка неопределенности через дисперсию весов
            # Это приблизительная оценка, основанная на дисперсии параметров
            if hasattr(br_model, 'sigma_') and br_model.sigma_ is not None:
                # Дисперсия предсказаний = X @ sigma_ @ X.T
                pred_var = np.diag(X_test_scaled @ br_model.sigma_ @ X_test_scaled.T)
                y_pred_std = np.sqrt(pred_var + 1.0 / br_model.alpha_)
            else:
                # Простая оценка через alpha (точность шума)
                y_pred_std = np.sqrt(1.0 / br_model.alpha_) * np.ones(len(X_test))
        except Exception as e:
            print(f"⚠️ Не удалось оценить неопределенность: {e}")
            y_pred_std = None
        
        # Метрики
        metrics = {
            'r2': r2_score(y_test, y_pred_mean),
            'mae': mean_absolute_error(y_test, y_pred_mean),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_mean))
        }
        
        # Сохранение результатов
        result = {
            'model': model,
            'metrics': metrics,
            'uncertainty': y_pred_std,
            'y_pred': y_pred_mean,
            'y_test': y_test.values,
            'features': valid_features,
            'n_samples': len(model_data),
            'alpha': br_model.alpha_ if hasattr(br_model, 'alpha_') else None,
            'lambda': br_model.lambda_ if hasattr(br_model, 'lambda_') else None
        }
        
        self.results['bayesian_ridge'] = result
        
        # Вывод результатов
        self._print_bayesian_ridge_results(result)
        
        return result
    
    def optimize_random_forest_bayesian(self, feature_columns: list,
                                       n_trials: int = 50,
                                       test_size: float = 0.2,
                                       random_state: int = 42,
                                       cv_folds: int = 5,
                                       use_optuna: bool = True) -> dict:
        """
        Байесовская оптимизация гиперпараметров Random Forest
        
        Parameters:
        -----------
        feature_columns : list
            Список колонок для использования в качестве признаков
        n_trials : int
            Количество попыток оптимизации
        test_size : float
            Доля тестовой выборки
        random_state : int
            Seed для воспроизводимости
        cv_folds : int
            Количество фолдов для кросс-валидации
        use_optuna : bool
            Использовать Optuna (True) или scikit-optimize (False)
            
        Returns:
        --------
        dict
            Словарь с результатами:
            - best_params: лучшие гиперпараметры
            - best_score: лучшее значение метрики
            - best_model: обученная модель с лучшими параметрами
            - metrics: метрики на тестовой выборке
            - study: объект study (для Optuna)
        """
        if not self.target:
            raise ValueError("Сначала установите целевую переменную с помощью set_target()")
        
        # Проверка доступности библиотек
        if use_optuna and not OPTUNA_AVAILABLE:
            print("⚠️ Optuna недоступна, используем scikit-optimize")
            use_optuna = False
        
        if not use_optuna and not SKOPT_AVAILABLE:
            print("⚠️ scikit-optimize не установлена. Устанавливаю Optuna как альтернативу...")
            if OPTUNA_AVAILABLE:
                use_optuna = True
            else:
                raise ValueError("Необходимо установить хотя бы одну библиотеку: pip install optuna или pip install scikit-optimize")
        
        # Подготовка данных
        valid_features = [f for f in feature_columns if f in self.data.columns]
        if len(valid_features) != len(feature_columns):
            missing = set(feature_columns) - set(valid_features)
            print(f"⚠️ Отсутствуют колонки: {missing}")
        
        model_data = self.data[valid_features + [self.target]].dropna()
        
        if len(model_data) < 50:
            raise ValueError(f"Слишком мало данных для обучения: {len(model_data)} строк")
        
        X = model_data[valid_features]
        y = model_data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if use_optuna:
            return self._optimize_with_optuna(
                X_train, y_train, X_test, y_test, 
                valid_features, n_trials, cv_folds, random_state
            )
        else:
            return self._optimize_with_skopt(
                X_train, y_train, X_test, y_test,
                valid_features, n_trials, cv_folds, random_state
            )
    
    def _optimize_with_optuna(self, X_train, y_train, X_test, y_test,
                              features, n_trials, cv_folds, random_state):
        """Оптимизация с помощью Optuna"""
        print(f"🔍 Начинаем байесовскую оптимизацию с Optuna ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        try:
            # Optuna 3.0+ использует show_progress_bar, старые версии - нет
            study.optimize(objective, n_trials=n_trials)
        except TypeError:
            # Для старых версий optuna
            study.optimize(objective, n_trials=n_trials)
        
        # Обучение лучшей модели
        best_params = study.best_params.copy()
        best_params['random_state'] = random_state
        best_params['n_jobs'] = -1
        
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        result = {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_model': best_model,
            'metrics': metrics,
            'study': study,
            'features': features,
            'y_pred': y_pred,
            'y_test': y_test.values
        }
        
        self.results['optimized_rf'] = result
        
        # Вывод результатов
        self._print_optimization_results(result, 'Optuna')
        
        return result
    
    def _optimize_with_skopt(self, X_train, y_train, X_test, y_test,
                             features, n_trials, cv_folds, random_state):
        """Оптимизация с помощью scikit-optimize"""
        print(f"🔍 Начинаем байесовскую оптимизацию с scikit-optimize ({n_trials} trials)...")
        
        # Определение пространства поиска
        space = [
            Integer(50, 500, name='n_estimators'),
            Integer(5, 30, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf'),
        ]
        
        @use_named_args(space=space)
        def objective(**params):
            params['random_state'] = random_state
            params['n_jobs'] = -1
            model = RandomForestRegressor(**params)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            return -scores.mean()
        
        result_gp = gp_minimize(
            objective, space, n_calls=n_trials,
            random_state=random_state,
            n_jobs=1,
            verbose=True
        )
        
        # Извлечение лучших параметров
        best_params = {
            'n_estimators': result_gp.x[0],
            'max_depth': result_gp.x[1],
            'min_samples_split': result_gp.x[2],
            'min_samples_leaf': result_gp.x[3],
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Обучение лучшей модели
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        result = {
            'best_params': best_params,
            'best_score': result_gp.fun,
            'best_model': best_model,
            'metrics': metrics,
            'optimization_result': result_gp,
            'features': features,
            'y_pred': y_pred,
            'y_test': y_test.values
        }
        
        self.results['optimized_rf'] = result
        
        # Вывод результатов
        self._print_optimization_results(result, 'scikit-optimize')
        
        return result
    
    def compare_bayesian_methods(self, feature_columns: list,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                n_trials: int = 30) -> pd.DataFrame:
        """
        Сравнение Bayesian Ridge и оптимизированного Random Forest
        
        Parameters:
        -----------
        feature_columns : list
            Список колонок для использования в качестве признаков
        test_size : float
            Доля тестовой выборки
        random_state : int
            Seed для воспроизводимости
        n_trials : int
            Количество попыток для оптимизации RF
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с результатами сравнения
        """
        print("=" * 70)
        print("🔬 СРАВНЕНИЕ БАЙЕСОВСКИХ МЕТОДОВ")
        print("=" * 70)
        
        results = []
        
        # 1. Bayesian Ridge
        print("\n1️⃣ Обучение Bayesian Ridge...")
        try:
            br_result = self.train_bayesian_ridge(
                feature_columns=feature_columns,
                test_size=test_size,
                random_state=random_state
            )
            results.append({
                'method': 'Bayesian Ridge',
                'r2': br_result['metrics']['r2'],
                'mae': br_result['metrics']['mae'],
                'rmse': br_result['metrics']['rmse'],
                'has_uncertainty': br_result['uncertainty'] is not None
            })
        except Exception as e:
            print(f"❌ Ошибка в Bayesian Ridge: {e}")
        
        # 2. Оптимизированный Random Forest
        print("\n2️⃣ Байесовская оптимизация Random Forest...")
        try:
            rf_result = self.optimize_random_forest_bayesian(
                feature_columns=feature_columns,
                n_trials=n_trials,
                test_size=test_size,
                random_state=random_state
            )
            results.append({
                'method': 'RF (Bayesian Optimized)',
                'r2': rf_result['metrics']['r2'],
                'mae': rf_result['metrics']['mae'],
                'rmse': rf_result['metrics']['rmse'],
                'best_params': str(rf_result['best_params']),
                'has_uncertainty': False
            })
        except Exception as e:
            print(f"❌ Ошибка в оптимизации RF: {e}")
        
        # 3. Базовый Random Forest для сравнения
        print("\n3️⃣ Базовый Random Forest (для сравнения)...")
        try:
            valid_features = [f for f in feature_columns if f in self.data.columns]
            model_data = self.data[valid_features + [self.target]].dropna()
            X = model_data[valid_features]
            y = model_data[self.target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            baseline_rf = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            )
            baseline_rf.fit(X_train, y_train)
            y_pred = baseline_rf.predict(X_test)
            
            results.append({
                'method': 'RF (Baseline)',
                'r2': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'has_uncertainty': False
            })
        except Exception as e:
            print(f"❌ Ошибка в базовом RF: {e}")
        
        # Формирование DataFrame и вывод
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 70)
        print("📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
        print("=" * 70)
        print(results_df.to_string(index=False))
        
        # Визуализация
        self._plot_bayesian_comparison(results_df)
        
        self.results['comparison'] = results_df
        return results_df
    
    def plot_uncertainty(self, result_key: str = 'bayesian_ridge', 
                        n_samples: int = 100):
        """
        Визуализация неопределенности предсказаний Bayesian Ridge
        
        Parameters:
        -----------
        result_key : str
            Ключ результата в self.results
        n_samples : int
            Количество случайных выборок для визуализации (если нужно)
        """
        if result_key not in self.results:
            print(f"❌ Результат '{result_key}' не найден")
            return
        
        result = self.results[result_key]
        
        if result.get('uncertainty') is None:
            print("⚠️ Оценка неопределенности недоступна")
            return
        
        y_test = result['y_test']
        y_pred = result['y_pred']
        y_std = result['uncertainty']
        
        # График предсказаний с интервалами неопределенности
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. Предсказания vs истинные значения
        axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Идеальная линия')
        axes[0].set_xlabel('Истинные значения', fontsize=12)
        axes[0].set_ylabel('Предсказанные значения', fontsize=12)
        axes[0].set_title('Предсказания Bayesian Ridge', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Остатки с интервалами неопределенности
        residuals = y_test - y_pred
        axes[1].errorbar(y_pred, residuals, yerr=y_std, 
                        fmt='o', alpha=0.5, markersize=4, capsize=2)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Предсказанные значения', fontsize=12)
        axes[1].set_ylabel('Остатки', fontsize=12)
        axes[1].set_title('Остатки с оценкой неопределенности', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _print_bayesian_ridge_results(self, result):
        """Вывод результатов Bayesian Ridge"""
        print("=" * 70)
        print("📊 РЕЗУЛЬТАТЫ BAYESIAN RIDGE")
        print("=" * 70)
        print(f"   R² = {result['metrics']['r2']:7.4f}")
        print(f"   MAE = {result['metrics']['mae']:7.4f}")
        print(f"   RMSE = {result['metrics']['rmse']:7.4f}")
        if result.get('alpha'):
            print(f"   Alpha (точность шума) = {result['alpha']:.6f}")
        if result.get('lambda'):
            print(f"   Lambda (регуляризация) = {result['lambda']:.6f}")
        if result['uncertainty'] is not None:
            print(f"   Средняя неопределенность = {result['uncertainty'].mean():.6f}")
        print(f"   Количество признаков: {len(result['features'])}")
        print(f"   Количество образцов: {result['n_samples']}")
    
    def _print_optimization_results(self, result, method_name: str):
        """Вывод результатов оптимизации"""
        print("=" * 70)
        print(f"📊 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ({method_name})")
        print("=" * 70)
        print(f"   Лучший score (CV): {result['best_score']:.6f}")
        print(f"   R² (test) = {result['metrics']['r2']:7.4f}")
        print(f"   MAE (test) = {result['metrics']['mae']:7.4f}")
        print(f"   RMSE (test) = {result['metrics']['rmse']:7.4f}")
        print(f"\n   Лучшие параметры:")
        for key, value in result['best_params'].items():
            if key not in ['random_state', 'n_jobs']:
                print(f"     {key}: {value}")
    
    def _plot_bayesian_comparison(self, results_df):
        """Визуализация сравнения байесовских методов"""
        if len(results_df) < 2:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['r2', 'mae', 'rmse']
        titles = ['R² (больше → лучше)', 'MAE (меньше → лучше)', 'RMSE (меньше → лучше)']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            axes[idx].bar(results_df['method'], results_df[metric], color=color)
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
            
            for i, v in enumerate(results_df[metric]):
                axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.suptitle('Сравнение байесовских методов', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

