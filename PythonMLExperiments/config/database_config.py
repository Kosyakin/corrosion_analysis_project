# Настройки подключения к базе данных
DATABASE_CONFIG = {
    'host': '100.98.56.203',
    'port': '5432',
    'database': 'asp_db',
    'user': 'homeuser',
    'password': '40000j,tpmzy'
}

CONNECTION_STRING = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"