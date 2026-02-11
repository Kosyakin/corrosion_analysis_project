# БД для показаний датчика SCD40/SCD41

## Таблица `air_samples`

Хранит показания датчиков качества воздуха: CO2 (ppm), температура (°C), влажность (%).

| Колонка       | Тип           | Описание |
|---------------|---------------|----------|
| id            | BIGSERIAL     | Первичный ключ |
| device_id     | TEXT          | Идентификатор устройства (например `esp32-lab-01`) |
| measured_at   | TIMESTAMPTZ   | Время измерения (UTC рекомендуется) |
| co2_ppm       | INTEGER       | Концентрация CO2, ppm (0–5000) |
| temperature_c | NUMERIC(5,2)  | Температура, °C |
| humidity_rh   | NUMERIC(5,2)  | Относительная влажность, % |
| fw_version    | TEXT          | Версия прошивки (опционально) |

## Создание таблицы

1. Откройте PostgreSQL (pgAdmin, DBeaver, psql и т.п.).
2. Подключитесь к нужной базе (или создайте новую: `CREATE DATABASE air_quality;`).
3. Выполните скрипт: [Tables/air_samples_create.sql](../Tables/air_samples_create.sql).

Через psql из папки проекта:

```bash
psql -U postgres -d your_database -f БД/Tables/air_samples_create.sql
```

## Пример запроса последних показаний

```sql
SELECT device_id, measured_at, co2_ppm, temperature_c, humidity_rh
FROM public.air_samples
ORDER BY measured_at DESC
LIMIT 100;
```

Дальше по плану: ASP.NET Core API принимает JSON от ESP32 и вставляет строки в эту таблицу.
