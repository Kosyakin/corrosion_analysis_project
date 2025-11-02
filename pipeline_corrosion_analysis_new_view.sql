CREATE OR REPLACE VIEW pipeline_corrosion_analysis_new_view AS
WITH operation_dates AS (
    SELECT DISTINCT ON (equipment_operation_dates.installation, equipment_operation_dates.equipment) 
        equipment_operation_dates.installation,
        equipment_operation_dates.equipment,
        equipment_operation_dates.start_date_of_operation
    FROM equipment_operation_dates
    ORDER BY equipment_operation_dates.installation, equipment_operation_dates.equipment, 
             equipment_operation_dates.start_date_of_operation DESC
), 
chemical_components AS (
    SELECT 
        equipment_chemical_environment.installation,
        equipment_chemical_environment.equipment,
        equipment_chemical_environment.contour,
        
        -- ОСНОВНЫЕ КОРРОЗИОННЫЕ АГЕНТЫ (High Priority)
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Вода' THEN equipment_chemical_environment.mol_percent END) AS water_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Сероводород' THEN equipment_chemical_environment.mol_percent END) AS h2s_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Хлор' THEN equipment_chemical_environment.mol_percent END) AS chlorine_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Диоксид углерода' THEN equipment_chemical_environment.mol_percent END) AS co2_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Кислород' THEN equipment_chemical_environment.mol_percent END) AS oxygen_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Соляная кислота' THEN equipment_chemical_environment.mol_percent END) AS hydrochloric_acid_content,
        
        -- КРИТИЧЕСКИЕ ТЕХНОЛОГИЧЕСКИЕ СРЕДЫ (Medium Priority)
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Аммиак (NH3)' THEN equipment_chemical_environment.mol_percent END) AS ammonia_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'AMMONIUM CHLORIDE' THEN equipment_chemical_environment.mol_percent END) AS ammonium_chloride_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Ингибитор коррозии' THEN equipment_chemical_environment.mol_percent END) AS corrosion_inhibitor_content,
        MAX(CASE WHEN equipment_chemical_environment.substance = 'Едкий натр' THEN equipment_chemical_environment.mol_percent END) AS sodium_hydroxide_content,
        
        -- УГЛЕВОДОРОДЫ (Low Priority - группируем)
        MAX(CASE WHEN equipment_chemical_environment.substance IN ('Метан', 'Этан', 'Пропан', 'Бутан') THEN equipment_chemical_environment.mol_percent END) AS light_hydrocarbons_content,
        MAX(CASE WHEN equipment_chemical_environment.substance IN ('Пентан', 'Изопентан', 'Гексан') THEN equipment_chemical_environment.mol_percent END) AS heavy_hydrocarbons_content,
        
        -- АГРЕГИРОВАННЫЕ ПОКАЗАТЕЛИ
        COUNT(*) AS total_components,
        SUM(equipment_chemical_environment.mol_percent) AS total_composition,
        
        -- КОМПОЗИТНЫЕ ИНДЕКСЫ КОРРОЗИИ
        COALESCE(MAX(CASE WHEN equipment_chemical_environment.substance IN ('Сероводород', 'Сера', 'Серная кислота', 'Этилмеркаптан') 
                     THEN equipment_chemical_environment.mol_percent END), 0) AS total_sulfur_compounds,
                     
        COALESCE(MAX(CASE WHEN equipment_chemical_environment.substance IN ('Соляная кислота', 'Хлор') 
                     THEN equipment_chemical_environment.mol_percent END), 0) AS total_chlorine_compounds,
                     
        COALESCE(MAX(CASE WHEN equipment_chemical_environment.substance IN ('Уксусная кислота', 'Нафтеновая кислота', 'Диоксид углерода') 
                     THEN equipment_chemical_environment.mol_percent END), 0) AS total_acids,
        
        -- ИНДЕКС АГРЕССИВНОСТИ СРЕДЫ (расчетный показатель)
        COALESCE(MAX(CASE WHEN equipment_chemical_environment.substance = 'Сероводород' THEN equipment_chemical_environment.mol_percent * 10 END), 0) +
        COALESCE(MAX(CASE WHEN equipment_chemical_environment.substance = 'Хлор' THEN equipment_chemical_environment.mol_percent * 5 END), 0) +
        COALESCE(MAX(CASE WHEN equipment_chemical_environment.substance = 'Соляная кислота' THEN equipment_chemical_environment.mol_percent * 8 END), 0) AS corrosion_aggressiveness_index
        
    FROM equipment_chemical_environment
    GROUP BY equipment_chemical_environment.installation, equipment_chemical_environment.equipment, equipment_chemical_environment.contour
),

component_types AS (
    SELECT 
        unique_components.component,
        -- Улучшенная логика классификации
        CASE 
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('УЧА', 'Уча', 'УЧ ', 'Уч ') THEN 1
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ОТВ', 'Отв', 'От4', 'От9', 'ОТ ', 'От ') THEN 2
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ТРО', 'Тр.', 'Тр(', 'Тро', 'ТР ', 'Тр ') THEN 3
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ПЕР', 'Пер', ' Пе', 'ПЕ ', 'Пер') THEN 4
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ЗАГ', 'Заг', 'ЗА ', 'За ') THEN 5
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ГИБ', 'Гиб', 'ГИ ', 'Ги ') THEN 6
            ELSE 0
        END AS component_type_id,
        
        CASE 
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('УЧА', 'Уча', 'УЧ ', 'Уч ') THEN 'Участок трубопровода'
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ОТВ', 'Отв', 'От4', 'От9', 'ОТ ', 'От ') THEN 'Отвод'
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ТРО', 'Тр.', 'Тр(', 'Тро', 'ТР ', 'Тр ') THEN 'Тройник'
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ПЕР', 'Пер', ' Пе', 'ПЕ ', 'Пер') THEN 'Переходник'
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ЗАГ', 'Заг', 'ЗА ', 'За ') THEN 'Заглушка'
            WHEN UPPER(LEFT(unique_components.component::text, 3)) IN ('ГИБ', 'Гиб', 'ГИ ', 'Ги ') THEN 'Гиб/Загиб'
            ELSE 'Неизвестный тип'
        END AS component_type_name
    FROM ( 
        SELECT DISTINCT measurements.component
        FROM measurements
    ) unique_components
),

measurement_sequence AS (
    SELECT 
        m.id,
        m.installation,
        m.equipment,
        m.mms,
        m.measurement_date,
        m.measurement,
        m.contour,
        od.start_date_of_operation,
        m.nominal_thickness_mmc,
        m.tmin_mmc,
        ec.operating_temperature,
        ec.operating_pressure,
        m.component,
        ec.material_code,
        ec.material_type,
        ec.outer_diameter,
        ec.inner_diameter,
        
        -- УЛУЧШЕННАЯ ОЦЕНКА КОРРОЗИОННОЙ СТОЙКОСТИ МАТЕРИАЛА
        CASE
            WHEN ec.material_type = 'Аустенитная сталь > 2.25% Mo' THEN 10
            WHEN ec.material_type = 'Аустенитная сталь (без Мо)' THEN 8
            WHEN ec.material_type = 'Сталь легированная Cr 5%' THEN 6
            WHEN ec.material_type = 'Сталь низколегированная' THEN 4
            WHEN ec.material_type = 'Углеродистая сталь 1 Cr 0.5 Mo' THEN 3
            WHEN ec.material_type = 'Углеродистая сталь' THEN 2
            ELSE 1
        END AS material_resistance_score,
        
        ec.outer_diameter - ec.inner_diameter AS wall_thickness,
        ec.outer_diameter / 2.0 AS radius,
        
        -- Расчет времени между измерениями
        LAG(m.measurement) OVER (PARTITION BY m.installation, m.equipment, m.mms ORDER BY m.measurement_date) AS previous_measurement,
        LAG(m.measurement_date) OVER (PARTITION BY m.installation, m.equipment, m.mms ORDER BY m.measurement_date) AS previous_measurement_date,
        
        -- Дополнительные технические параметры
        EXTRACT(YEAR FROM m.measurement_date) - EXTRACT(YEAR FROM od.start_date_of_operation) AS equipment_age_years
        
    FROM measurements m
    LEFT JOIN equipment_components ec ON m.installation::text = ec.installation::text 
                                     AND m.equipment::text = ec.equipment::text 
                                     AND m.component::text = ec.component::text
    LEFT JOIN operation_dates od ON m.installation::text = od.installation::text 
                                 AND m.equipment::text = od.equipment::text
    WHERE 
        ec.component_type::text ~~ 'Участок трубопровода'::text
        AND (m.equipment::text ~~ '%Т-%'::text OR m.equipment::text ~~ '%T-%'::text)
        AND (ec.source_component IS NOT NULL OR ec.material_code IS NOT NULL OR ec.material_type IS NOT NULL)
)

SELECT 
    ms.id,
    ms.installation,
    ms.equipment,
    ms.mms,
    ms.measurement_date,
    ms.measurement,
    
    -- ФЛАГ ЗАМЕНЫ ОБОРУДОВАНИЯ
    CASE
        WHEN ms.measurement = ms.nominal_thickness_mmc 
             AND (ms.previous_measurement IS NULL OR ms.previous_measurement <> ms.nominal_thickness_mmc) 
        THEN true
        ELSE false
    END AS is_replaced,
    
    -- РАСЧЕТ СКОРОСТИ КОРРОЗИИ (мм/год)
    CASE
        WHEN ms.previous_measurement IS NOT NULL 
             AND ms.previous_measurement_date IS NOT NULL 
             AND ms.measurement_date > ms.previous_measurement_date 
        THEN
            CASE
                WHEN ms.previous_measurement = ms.measurement THEN 0.0
                WHEN ms.previous_measurement > ms.measurement THEN 
                    (ms.previous_measurement - ms.measurement) / 
                    GREATEST(EXTRACT(DAY FROM (ms.measurement_date - ms.previous_measurement_date)) / 365.25, 0.00274) -- минимум 1 день
                ELSE 0.0
            END
        ELSE NULL::numeric
    END AS corrosion_rate_mm_per_year,
    
    -- ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ КОРРОЗИИ
    CASE 
        WHEN ms.previous_measurement IS NOT NULL AND ms.measurement < ms.previous_measurement 
        THEN (ms.previous_measurement - ms.measurement) / ms.previous_measurement * 100 
        ELSE 0 
    END AS wall_loss_percentage,
    
    -- ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ
    ct.component_type_id,
    ct.component_type_name,
    ms.wall_thickness,
    ms.radius,
    ms.outer_diameter,
    ms.inner_diameter,
    
    -- ГЕОМЕТРИЧЕСКИЕ ХАРАКТЕРИСТИКИ
    CASE
        WHEN ms.wall_thickness > 0::numeric THEN ms.outer_diameter / ms.wall_thickness
        ELSE NULL::numeric
    END AS diameter_to_thickness_ratio,
    
    PI() * POWER(ms.outer_diameter / 2.0, 2) - PI() * POWER(ms.inner_diameter / 2.0, 2) AS cross_sectional_area,
    
    -- КОРРОЗИОННЫЕ ПАРАМЕТРЫ
    cc.corrosion_aggressiveness_index,
    cc.total_sulfur_compounds,
    cc.total_chlorine_compounds,
    cc.total_acids,
    
    -- ОСНОВНЫЕ ХИМИЧЕСКИЕ КОМПОНЕНТЫ
    cc.water_content,
    cc.h2s_content,
    cc.chlorine_content,
    cc.co2_content,
    cc.oxygen_content,
    cc.hydrochloric_acid_content,
    cc.ammonia_content,
    cc.corrosion_inhibitor_content,
    
    -- СЛУЖЕБНЫЕ ПОЛЯ
    ms.start_date_of_operation,
    ms.equipment_age_years,
    ms.nominal_thickness_mmc,
    ms.tmin_mmc,
    ms.contour,
    ms.operating_temperature,
    ms.operating_pressure,
    ms.component,
    ms.material_code,
    ms.material_type,
    ms.material_resistance_score
    
FROM measurement_sequence ms
LEFT JOIN chemical_components cc ON ms.installation::text = cc.installation::text 
                                 AND ms.equipment::text = cc.equipment::text 
                                 AND ms.contour::text = cc.contour::text
LEFT JOIN component_types ct ON ms.component::text = ct.component::text
ORDER BY ms.installation, ms.equipment, ms.mms, ms.measurement_date;