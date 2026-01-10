-- Этот файл выполнится автоматически при первом запуске контейнера PostgreSQL
-- Устанавливаем параметры для Redmine
ALTER DATABASE redmine SET client_encoding = 'UTF8';
ALTER DATABASE redmine SET default_transaction_isolation = 'read committed';
ALTER DATABASE redmine SET timezone = 'UTC';

-- Создаем расширение для полнотекстового поиска (опционально)
CREATE EXTENSION IF NOT EXISTS pg_trgm;