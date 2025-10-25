CREATE OR REPLACE VIEW pipeline_corrosion_analysis_view AS
WITH operation_dates AS (
    SELECT DISTINCT ON (equipment_operation_dates.installation, equipment_operation_dates.equipment) 
        equipment_operation_dates.installation,
        equipment_operation_dates.equipment,
        equipment_operation_dates.start_date_of_operation
    FROM equipment_operation_dates
    ORDER BY equipment_operation_dates.installation, equipment_operation_dates.equipment, equipment_operation_dates.start_date_of_operation DESC
), 
chemical_components AS (
    SELECT 
        "equipment_сhemical_environment".installation,
        "equipment_сhemical_environment".equipment,
        "equipment_сhemical_environment".contour,
        -- Основные коррозионно-активные компоненты
        MAX(CASE WHEN substance = 'Вода' THEN mol_percent END) AS water_content,
        MAX(CASE WHEN substance = 'Сероводород' THEN mol_percent END) AS h2s_content,
        MAX(CASE WHEN substance = 'Сера' THEN mol_percent END) AS sulfur_content,
        MAX(CASE WHEN substance = 'Хлор' THEN mol_percent END) AS chlorine_content,
        MAX(CASE WHEN substance = 'Диоксид углерода' THEN mol_percent END) AS co2_content,
        MAX(CASE WHEN substance = 'Кислород' THEN mol_percent END) AS oxygen_content,
        MAX(CASE WHEN substance = 'Азот' THEN mol_percent END) AS nitrogen_content,
        MAX(CASE WHEN substance = 'Водород' THEN mol_percent END) AS hydrogen_content,
        
        -- Углеводороды легкие
        MAX(CASE WHEN substance = 'Метан' THEN mol_percent END) AS methane_content,
        MAX(CASE WHEN substance = 'Этан' THEN mol_percent END) AS ethane_content,
        MAX(CASE WHEN substance = 'Пропан' THEN mol_percent END) AS propane_content,
        MAX(CASE WHEN substance = 'Бутан' THEN mol_percent END) AS butane_content,
        MAX(CASE WHEN substance = 'Изобутан' THEN mol_percent END) AS isobutane_content,
        
        -- Углеводороды C5-C8
        MAX(CASE WHEN substance = 'Пентан' THEN mol_percent END) AS pentane_content,
        MAX(CASE WHEN substance = 'Изопентан' THEN mol_percent END) AS isopentane_content,
        MAX(CASE WHEN substance = 'C6-C8 (Бензин)' THEN mol_percent END) AS gasoline_c6_c8_content,
        MAX(CASE WHEN substance = 'Гексан' THEN mol_percent END) AS hexane_content,
        
        -- Углеводороды C9-C16
        MAX(CASE WHEN substance = 'C9-C12 (Тяжёлая нафта)' THEN mol_percent END) AS heavy_naphtha_content,
        MAX(CASE WHEN substance = 'C13-C16 (Керосин)' THEN mol_percent END) AS kerosene_content,
        
        -- Углеводороды C17+
        MAX(CASE WHEN substance = 'C17-C25 (Дизель-Газойль)' THEN mol_percent END) AS diesel_content,
        MAX(CASE WHEN substance = 'C25+ (Остатки)' THEN mol_percent END) AS residues_content,
        
        -- Непредельные углеводороды
        MAX(CASE WHEN substance = 'Пропилен' THEN mol_percent END) AS propylene_content,
        MAX(CASE WHEN substance = 'Этилен' THEN mol_percent END) AS ethylene_content,
        MAX(CASE WHEN substance = 'Бутилен' THEN mol_percent END) AS butylene_content,
        
        -- Кислоты
        MAX(CASE WHEN substance = 'Серная кислота' THEN mol_percent END) AS sulfuric_acid_content,
        MAX(CASE WHEN substance = 'Соляная кислота' THEN mol_percent END) AS hydrochloric_acid_content,
        MAX(CASE WHEN substance = 'Уксусная кислота' THEN mol_percent END) AS acetic_acid_content,
        MAX(CASE WHEN substance = 'Нафтеновая кислота' THEN mol_percent END) AS naphthenic_acid_content,
        
        -- Прочие компоненты
        MAX(CASE WHEN substance = 'Аммиак' THEN mol_percent END) AS ammonia_content,
        MAX(CASE WHEN substance = 'Аммоний' THEN mol_percent END) AS ammonium_content,
        MAX(CASE WHEN substance = 'Фтористый водород' THEN mol_percent END) AS hydrogen_fluoride_content,
        MAX(CASE WHEN substance = 'Едкий натр' THEN mol_percent END) AS sodium_hydroxide_content,
        MAX(CASE WHEN substance = 'Ингибитор коррозии' THEN mol_percent END) AS corrosion_inhibitor_content,
        
        -- Агрегатные показатели
        COUNT(*) as total_components,
        SUM(mol_percent) as total_composition,
        MAX(CASE WHEN substance IN ('Сероводород', 'Сера', 'Серная кислота') THEN mol_percent ELSE 0 END) AS total_sulfur_compounds,
        MAX(CASE WHEN substance IN ('Соляная кислота', 'Хлор') THEN mol_percent ELSE 0 END) AS total_chlorine_compounds,
        MAX(CASE WHEN substance IN ('Уксусная кислота', 'Нафтеновая кислота') THEN mol_percent ELSE 0 END) AS total_acids
    FROM "equipment_сhemical_environment"
    GROUP BY installation, equipment, contour
),
-- CTE для определения геометрических типов компонентов
component_types AS (
    SELECT 
        component,
        CASE 
            WHEN UPPER(LEFT(component, 3)) IN ('УЧА', 'Уча') THEN 1  -- Участок
            WHEN UPPER(LEFT(component, 3)) IN ('ОТВ', 'Отв', 'От4', 'От9') THEN 2  -- Отвод
            WHEN UPPER(LEFT(component, 3)) IN ('ТРО', 'Тр.', 'Тр(', 'Тро') THEN 3  -- Тройник
            WHEN UPPER(LEFT(component, 3)) IN ('ПЕР', 'Пер', ' Пе') THEN 4  -- Переходник
            WHEN UPPER(LEFT(component, 3)) IN ('ЗАГ', 'Заг') THEN 5  -- Заглушка
            WHEN UPPER(LEFT(component, 3)) IN ('ГИБ', 'Гиб') THEN 6  -- Гиб/Загиб
            ELSE 0  -- Неизвестный тип
        END AS component_type_id,
        CASE 
            WHEN UPPER(LEFT(component, 3)) IN ('УЧА', 'Уча') THEN 'Участок'
            WHEN UPPER(LEFT(component, 3)) IN ('ОТВ', 'Отв', 'От4', 'От9') THEN 'Отвод'
            WHEN UPPER(LEFT(component, 3)) IN ('ТРО', 'Тр.', 'Тр(', 'Тро') THEN 'Тройник'
            WHEN UPPER(LEFT(component, 3)) IN ('ПЕР', 'Пер', ' Пе') THEN 'Переходник'
            WHEN UPPER(LEFT(component, 3)) IN ('ЗАГ', 'Заг') THEN 'Заглушка'
            WHEN UPPER(LEFT(component, 3)) IN ('ГИБ', 'Гиб') THEN 'Гиб/Загиб'
            ELSE 'Неизвестный'
        END AS component_type_name
    FROM (SELECT DISTINCT component FROM measurements) AS unique_components
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
        -- Геометрические параметры
        ec.outer_diameter - ec.inner_diameter AS wall_thickness,
        ec.outer_diameter / 2.0 AS radius,
        -- Предыдущее измерение
        LAG(m.measurement) OVER (PARTITION BY m.installation, m.equipment, m.mms ORDER BY m.measurement_date) AS previous_measurement,
        LAG(m.measurement_date) OVER (PARTITION BY m.installation, m.equipment, m.mms ORDER BY m.measurement_date) AS previous_measurement_date
    FROM measurements m
    LEFT JOIN equipment_components ec ON 
        m.installation::text = ec.installation::text 
        AND m.equipment::text = ec.equipment::text 
        AND m.component::text = ec.component::text
    LEFT JOIN operation_dates od ON
        m.installation::text = od.installation::text 
        AND m.equipment::text = od.equipment::text
    WHERE 
        ec.component_type LIKE 'Участок трубопровода'
        AND (m.equipment LIKE '%Т-%' OR m.equipment LIKE '%T-%')
        AND (
            ec.source_component IS NOT NULL OR
            ec.material_code IS NOT NULL OR
            ec.material_type IS NOT NULL OR
            ec.component_type IS NOT NULL OR
            ec.operating_temperature IS NOT NULL OR
            ec.operating_pressure IS NOT NULL OR
            ec.outer_diameter IS NOT NULL OR
            ec.inner_diameter IS NOT NULL OR
            ec.initial_thickness IS NOT NULL
        )
)
SELECT 
    -- Основные данные измерений
    ms.id,
    ms.installation,
    ms.equipment,
    ms.mms,
    ms.measurement_date,
    ms.measurement,
    
    -- Индикатор замены
    CASE 
        WHEN ms.measurement = ms.nominal_thickness_mmc 
            AND (ms.previous_measurement IS NULL OR ms.previous_measurement != ms.nominal_thickness_mmc)
        THEN true
        ELSE false
    END AS is_replaced,
    
    -- Скорость коррозии
    CASE 
        WHEN ms.previous_measurement IS NOT NULL 
            AND ms.previous_measurement_date IS NOT NULL
            AND ms.measurement_date > ms.previous_measurement_date
        THEN 
            CASE 
                WHEN ms.previous_measurement = ms.measurement THEN 0.0
                WHEN ms.previous_measurement > ms.measurement THEN 
                    (ms.previous_measurement - ms.measurement) / 
                    GREATEST((ms.measurement_date - ms.previous_measurement_date)::numeric, 1) * 365.25
                ELSE 0.0
            END
        ELSE NULL 
    END AS corrosion_rate,
    
    -- Геометрические признаки
    ct.component_type_id,
    ct.component_type_name,
    ms.wall_thickness,
    ms.radius,
    ms.outer_diameter,
    ms.inner_diameter,
    -- Отношение диаметра к толщине стенки (показатель гибкости)
    CASE 
        WHEN ms.wall_thickness > 0 THEN ms.outer_diameter / ms.wall_thickness 
        ELSE NULL 
    END AS diameter_to_thickness_ratio,
    -- Площадь поперечного сечения
    (PI() * (ms.outer_diameter/2.0)^2 - PI() * (ms.inner_diameter/2.0)^2) AS cross_sectional_area,
    
    -- Химический состав среды
    cc.water_content,
    cc.h2s_content,
    cc.sulfur_content,
    cc.chlorine_content,
    cc.co2_content,
    cc.oxygen_content,
    cc.nitrogen_content,
    cc.hydrogen_content,
    cc.methane_content,
    cc.ethane_content,
    cc.propane_content,
    cc.butane_content,
    cc.isobutane_content,
    cc.pentane_content,
    cc.isopentane_content,
    cc.gasoline_c6_c8_content,
    cc.hexane_content,
    cc.heavy_naphtha_content,
    cc.kerosene_content,
    cc.diesel_content,
    cc.residues_content,
    cc.propylene_content,
    cc.ethylene_content,
    cc.butylene_content,
    cc.sulfuric_acid_content,
    cc.hydrochloric_acid_content,
    cc.acetic_acid_content,
    cc.naphthenic_acid_content,
    cc.ammonia_content,
    cc.ammonium_content,
    cc.hydrogen_fluoride_content,
    cc.sodium_hydroxide_content,
    cc.corrosion_inhibitor_content,
    cc.total_components,
    cc.total_composition,
    cc.total_sulfur_compounds,
    cc.total_chlorine_compounds,
    cc.total_acids,
    
    -- Конструкционные параметры
    ms.start_date_of_operation,
    ms.nominal_thickness_mmc,
    ms.tmin_mmc,
    ms.contour,
    ms.operating_temperature,
    ms.operating_pressure,
    ms.component,
    ms.material_code,
    ms.material_type

FROM measurement_sequence ms
LEFT JOIN chemical_components cc ON 
    ms.installation::text = cc.installation::text 
    AND ms.equipment::text = cc.equipment::text 
    AND ms.contour::text = cc.contour::text
LEFT JOIN component_types ct ON 
    ms.component::text = ct.component::text
ORDER BY ms.installation, ms.equipment, ms.mms, ms.measurement_date;