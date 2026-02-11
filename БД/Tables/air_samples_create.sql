-- Таблица показаний датчика SCD40/SCD41.

BEGIN;

DROP TABLE IF EXISTS public.air_samples CASCADE;

CREATE TABLE public.air_samples (
    id              BIGSERIAL PRIMARY KEY,
    device_id       TEXT NOT NULL,
    measured_at      TIMESTAMPTZ NOT NULL,
    co2_ppm         INTEGER NOT NULL,
    temperature_c   NUMERIC(5, 2) NOT NULL,
    humidity_rh      NUMERIC(5, 2) NOT NULL,
    fw_version      TEXT NULL,
    CONSTRAINT chk_co2_ppm CHECK (co2_ppm >= 0 AND co2_ppm <= 5000),
    CONSTRAINT chk_temperature CHECK (temperature_c >= -40 AND temperature_c <= 85),
    CONSTRAINT chk_humidity CHECK (humidity_rh >= 0 AND humidity_rh <= 100)
);

COMMIT;
