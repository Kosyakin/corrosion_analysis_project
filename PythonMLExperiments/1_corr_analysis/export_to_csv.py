"""
Выгрузка исходного датасета из БД в CSV.
Запуск из папки 1_corr_analysis: python export_to_csv.py
"""
import os
import sys

if __name__ == "__main__":
    if "../src" not in sys.path:
        sys.path.append("../src")
    from database import load_worst_corrosion_by_component as load_data

    print("Загрузка данных из БД...")
    df = load_data()
    out_path = os.path.join(os.path.dirname(__file__), "worst_corrosion_by_component.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {out_path}")
    print(f"Строк: {len(df):,}, колонок: {len(df.columns)}")
