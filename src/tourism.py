"""
Функции загрузки и предобработки данных о туризме в регионах России.

Используются в ноутбуке notebooks/tourism_analysis.ipynb.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = {
    "hotel": "hotel.csv",
    "person_hotel": "count_person_hotel.csv",
    "tour_firm": "tour_firm.csv",
    "tour_cost": "tour_cost_2023.csv",
}

# Префиксы исходных названий столбцов -> короткие имена
HOTEL_PREFIXES = {
    "Число гостиниц": "hotels",
    "Число мест в гостиницах": "beds_hotels",
    "Число хостелов": "hostels",
    "Число мест в хостелах": "beds_hostels",
    "Число санаторно-курортных организаций": "spa_orgs",
    "Число мест в санаторно-курортных организациях": "beds_spa",
}

TOUR_FIRM_PREFIXES = {
    "Общее число турпакетов, реализованных населению": "total",
    "Общее число турпакетов, реализованных гражданам России по территории России": "domestic",
    "Общее число турпакетов, реализованных гражданам России по другим странам": "outbound",
    "Общее число турпакетов, реализованных гражданам других стран по территории России": "inbound",
}

PERSON_HOTEL_COLUMNS = ["region", "guests_total_2023", "guests_russian_2023", "guests_foreign_2023"]
TOUR_COST_COLUMNS = ["region", "cost_total_2023", "cost_domestic_2023", "cost_outbound_2023"]

# Обозначения Росстата, которые означают «нет данных», а не ноль.
# «…1» — сноска: данные не публикуются в целях обеспечения конфиденциальности
# первичных статистических данных (ФЗ № 282-ФЗ).
MISSING_TOKENS = {"…1", "...1", "…", "...", "-", "—", ""}

AGGREGATE_PATTERN = re.compile(r"российская федерация|федеральный округ", re.IGNORECASE)

# Росстат иногда даёт «матрёшку»: область целиком, входящие в неё автономные округа
# и область без них. В остальных таблицах области даны целиком, поэтому оставляем
# только строку «область целиком», а вложенные строки убираем, чтобы не считать дважды.
NESTED_PATTERN = re.compile(r"\(кроме |\bбез автономн", re.IGNORECASE)
NESTED_AUTONOMOUS = {
    "Ненецкий автономный округ",
    "Ханты-Мансийский автономный округ - Югра",
    "Ямало-Ненецкий автономный округ",
}


def _normalize(text: str) -> str:
    """Убирает лишние пробелы, в том числе двойные внутри строки."""
    return " ".join(str(text).split())


def normalize_region(name: str) -> str:
    """Приводит названия к одному виду: «г.Москва» -> «г. Москва»."""
    name = _normalize(name)
    return re.sub(r"^г\.\s*", "г. ", name)


def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Читает файлы как строки, чтобы ничего не потерять при разборе.
    Отсутствующий файл пропускается с предупреждением, а не роняет весь анализ.
    """
    tables = {}
    for name, file in FILES.items():
        path = raw_dir / file
        if not path.exists():
            print(f"[!] Файл {file} не найден в data/raw — разделы с ним будут пропущены")
            continue
        tables[name] = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")
    return tables


def rename_columns(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Переименовывает столбцы во всех загруженных таблицах."""
    renamers = {
        "hotel": lambda df: rename_by_prefix(df, HOTEL_PREFIXES),
        "person_hotel": lambda df: rename_positional(df, PERSON_HOTEL_COLUMNS),
        "tour_firm": lambda df: rename_by_prefix(df, TOUR_FIRM_PREFIXES),
        "tour_cost": lambda df: rename_positional(df, TOUR_COST_COLUMNS),
    }
    return {name: renamers[name](df) for name, df in raw.items()}


def rename_by_prefix(df: pd.DataFrame, prefixes: dict[str, str]) -> pd.DataFrame:
    """
    Переименовывает столбцы вида «<показатель>, <год>» в «<короткое_имя>_<год>».
    Первый столбец всегда «Субъект» -> region.
    """
    mapping = {df.columns[0]: "region"}
    for col in df.columns[1:]:
        name, _, year = _normalize(col).rpartition(", ")
        if name not in prefixes or not year.isdigit():
            raise ValueError(f"Неизвестный столбец: {col!r}")
        mapping[col] = f"{prefixes[name]}_{year}"
    return df.rename(columns=mapping)


def rename_positional(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Переименовывает столбцы по порядку с проверкой их числа."""
    if len(df.columns) != len(columns):
        raise ValueError(f"Ожидалось {len(columns)} столбцов, получено {len(df.columns)}")
    out = df.copy()
    out.columns = columns
    return out


def find_non_numeric(df: pd.DataFrame) -> pd.Series:
    """Значения, которые не удаётся прочитать как число, и сколько раз они встречаются."""
    values = df.drop(columns="region").stack().dropna().map(_normalize)
    cleaned = values.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    bad = values[pd.to_numeric(cleaned, errors="coerce").isna()]
    return bad.value_counts()


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит все столбцы, кроме region, к float.
    Обозначения «нет данных» превращаются в NaN, а не в 0:
    ноль занизил бы средние и исказил статистические тесты.
    """
    out = df.copy()
    out["region"] = out["region"].map(normalize_region)
    for col in out.columns.drop("region"):
        s = out[col].map(lambda v: np.nan if pd.isna(v) else _normalize(v))
        s = s.where(~s.isin(MISSING_TOKENS))
        s = s.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        out[col] = pd.to_numeric(s, errors="coerce")
    return out


def split_aggregates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Отделяет от субъектов итоговые и вложенные строки:
    РФ и федеральные округа, а также автономные округа в составе областей
    и строки вида «область (кроме …)» / «… без автономных округов».
    Если их оставить, они попадут в рейтинги и будут посчитаны дважды.
    """
    region = df["region"]
    mask = (
        region.str.contains(AGGREGATE_PATTERN, na=False)
        | region.str.contains(NESTED_PATTERN, na=False)
        | region.isin(NESTED_AUTONOMOUS)
    )
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по столбцам: тип, пропуски, доля пропусков, уникальные значения."""
    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_share": df.isna().mean().round(3),
        "unique": df.nunique(),
    })
    return report.sort_values("missing", ascending=False)


def wide_to_long(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    """Таблица «регион × показатель_год» -> длинный формат region / year / показатели."""
    stub = pd.wide_to_long(df, stubnames=prefixes, i="region", j="year", sep="_")
    return stub.reset_index()


def prepare_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Полный конвейер предобработки: загрузка -> переименование -> числа -> без агрегатов."""
    renamed = rename_columns(load_raw(raw_dir))
    clean = {}
    for name, df in renamed.items():
        regions, _ = split_aggregates(to_numeric(df))
        clean[name] = regions.drop_duplicates(subset="region")
    return clean
