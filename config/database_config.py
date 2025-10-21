# Настройки подключения к базе данных
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'asp_db',
    'user': 'admin',
    'password': 'ASDqwe123'
}

CONNECTION_STRING = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"