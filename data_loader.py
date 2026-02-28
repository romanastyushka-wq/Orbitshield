import pandas as pd

COLUMNS = [
    "year", "day", "hour",
    "IMF_Magnitude",
    "Bx", "By", "Bz",
    "Proton_Density",
    "Flow_Speed",
    "Kp_x10"
]

def load_txt_data(path):
    # Читаем с обработкой ошибок парсинга
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=COLUMNS,
        on_bad_lines='skip'
    )

    # Создаем datetime индекс
    df["datetime"] = pd.to_datetime(
        df["year"].astype(str), format="%Y"
    ) + pd.to_timedelta(df["day"] - 1, unit="D") \
      + pd.to_timedelta(df["hour"], unit="h")

    # Сортировка обязательна для Time Series
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Удаляем дубликаты по времени
    df = df.drop_duplicates(subset=["datetime"])

    return df