import pandas as pd
from sqlalchemy import create_engine
import os
import sys

# Add the project root to Python path to find config module
try:
    # When running as a module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
except NameError:
    # When running from notebook or interactive session
    project_root = os.path.dirname(os.getcwd())

if project_root not in sys.path:
    sys.path.append(project_root)

# Try different import methods
try:
    from config.database_config import CONNECTION_STRING
except ImportError:
    # Fallback: direct import from file
    config_path = os.path.join(project_root, 'config', 'database_config.py')
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("database_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        CONNECTION_STRING = config_module.CONNECTION_STRING
    else:
        raise ImportError(f"Could not find config file at {config_path}")

def load_corrosion_data():
    """Загрузка данных из представления"""
    engine = create_engine(CONNECTION_STRING)
    query = "SELECT * FROM pipeline_corrosion_analysis_view"
    df = pd.read_sql(query, engine)
    return df

def load_raw_data():
    """Загрузка сырых данных для глубокого анализа"""
    engine = create_engine(CONNECTION_STRING)
    query = """
    SELECT * FROM measurements m
    LEFT JOIN equipment_components ec ON m.installation = ec.installation 
        AND m.equipment = ec.equipment 
        AND m.component = ec.component
    WHERE m.equipment LIKE '%Т-%' OR m.equipment LIKE '%T-%'
    """
    df = pd.read_sql(query, engine)
    return df