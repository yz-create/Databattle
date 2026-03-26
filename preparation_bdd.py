import pandas as pd
import numpy as np

df = pd.read_csv("segment_alerts_all_airports_train.csv")
print(df.count())
# vérifier les colonnes qui finissent par "00:00"
df['date'].str.endswith("00:00").all()

df['date'] = df['date'].str[:-6]
df['date'] = pd.to_datetime(df['date'])

# Création d'un id par alerte
df_clean = df[df["airport_alert_id"].notna() & (df["airport_alert_id"] != "")].copy()

df_clean["is_last_lightning_cloud_ground"] = (
    df_clean["is_last_lightning_cloud_ground"]
    .fillna(False)
    .astype(bool)
)

df_clean = df_clean.sort_values(["airport", "date"]).copy()

df_clean["id_eclair"] = (
    df_clean.groupby("airport")["is_last_lightning_cloud_ground"]
    .transform(lambda s: s.shift(fill_value=False).cumsum() + 1)
)

df_id_orages = df.copy()

df_id_orages = df_id_orages.sort_values(["airport", "date"]).reset_index(drop=True)

df_id_orages["is_last_lightning_cloud_ground"] = (
    df_id_orages["is_last_lightning_cloud_ground"]
    .replace({"True": True, "False": False, True: True, False: False})
    .fillna(False)
    .astype(bool)
)

df_id_orages["id_orage"] = (
    df_id_orages["is_last_lightning_cloud_ground"]
    .shift(fill_value=False)
    .cumsum()
    + 1
)

prep = (
    df_id_orages[
        df_id_orages["airport_alert_id"].notna()
        & (df_id_orages["airport_alert_id"] != "")
    ]
    .groupby("id_orage", as_index=False)
    .agg(
        premier_eclairns_20km=("date", "min"),
        dernier_eclairns_20km=("date", "max"),
    )
    .drop_duplicates()
)

df_premier_dernier_eclair = df_id_orages.merge(prep, on="id_orage", how="left")

df_premier_dernier_eclair["icloud"] = (
    df_premier_dernier_eclair["icloud"]
    .replace({"False": False, "True": True, False: False, True: True})
)

df_ratio = df_premier_dernier_eclair.copy()

df_ratio["date"] = pd.to_datetime(df_ratio["date"])
df_ratio = df_ratio.sort_values(["airport", "date"]).reset_index(drop=True)
df_ratio["ic"] = df_ratio["icloud"].astype(int)


def compute_ratio_fast(df, start_min, end_min):
    result = np.zeros(len(df), dtype=float)

    for airport, idx in df.groupby("airport").groups.items():
        group = df.loc[idx].sort_values("date")
        dates = group["date"].values.astype("datetime64[ns]")
        ic = group["ic"].values

        vals = np.zeros(len(group), dtype=float)

        for i, current_date in enumerate(dates):
            start = current_date - np.timedelta64(end_min, "m")
            end = current_date - np.timedelta64(start_min, "m")

            mask = (dates >= start) & (dates < end)
            n = mask.sum()

            vals[i] = ic[mask].sum() / n if n > 0 else 0.0

        result[group.index] = vals

    return result


df_ratio["ratio_ic_0_5"] = compute_ratio_fast(df_ratio, 0, 5)
df_ratio["ratio_ic_5_10"] = compute_ratio_fast(df_ratio, 5, 10)
df_ratio["ratio_ic_10_20"] = compute_ratio_fast(df_ratio, 10, 20)
df_ratio["ratio_ic_20_120"] = compute_ratio_fast(df_ratio, 20, 120)

dt = df_premier_dernier_eclair.copy()

dt["date"] = pd.to_datetime(dt["date"])
dt = dt.sort_values(["airport", "date"]).reset_index(drop=True)

# bool -> int
dt["icloud"] = dt["icloud"].fillna(False).astype(bool)
dt["ic"] = dt["icloud"].astype(int)
dt["cg"] = (~dt["icloud"]).astype(int)

# indicateurs de distance
dt["d_0_20"] = (dt["dist"] <= 20).astype(int)
dt["d_20_30"] = ((dt["dist"] > 20) & (dt["dist"] <= 30)).astype(int)


def add_count_windows_fast(df, value_col, suffix):
    out = []

    for airport, g in df.groupby("airport", sort=False):
        g = g.sort_values("date").copy()
        g = g.set_index("date")

        # cumuls sur fenêtres glissantes
        r5 = g[value_col].rolling("5min", closed="left").sum()
        r10 = g[value_col].rolling("10min", closed="left").sum()
        r20 = g[value_col].rolling("20min", closed="left").sum()
        r120 = g[value_col].rolling("120min", closed="left").sum()

        g[f"n_0_5_{suffix}"] = r5
        g[f"n_5_10_{suffix}"] = r10 - r5
        g[f"n_10_20_{suffix}"] = r20 - r10
        g[f"n_20_120_{suffix}"] = r120 - r20

        out.append(g.reset_index())

    return pd.concat(out, ignore_index=True)


dt_0_20 = add_count_windows_fast(dt, "d_0_20", "0_20")
dt_20_30 = add_count_windows_fast(dt, "d_20_30", "20_30")

cols_0_20 = [
    "lightning_id",
    "n_0_5_0_20",
    "n_5_10_0_20",
    "n_10_20_0_20",
    "n_20_120_0_20",
]

cols_20_30 = [
    "lightning_id",
    "n_0_5_20_30",
    "n_5_10_20_30",
    "n_10_20_20_30",
    "n_20_120_20_30",
]

df_ratio_fast = df_ratio.copy()

df_ratio_fast = df_ratio_fast.merge(dt_0_20[cols_0_20], on="lightning_id", how="left")
df_ratio_fast = df_ratio_fast.merge(dt_20_30[cols_20_30], on="lightning_id", how="left")

count_cols = [
    "n_0_5_0_20",
    "n_5_10_0_20",
    "n_10_20_0_20",
    "n_20_120_0_20",
    "n_0_5_20_30",
    "n_5_10_20_30",
    "n_10_20_20_30",
    "n_20_120_20_30",
]

df_ratio_fast[count_cols] = (
    df_ratio_fast[count_cols]
    .fillna(0)
    .round()
    .astype(int)
)


print(df_ratio_fast.columns)

db_date = df_ratio_fast[['id_orage', 'date', 'lightning_id']]
db_date['time_before_end'] = np.nan
db_date['date'] = pd.to_datetime(db_date['date'])
db_date = db_date.sort_values(by=['id_orage', 'date'])
last_lightning = db_date.groupby('id_orage')['date'].max()
db_date = db_date.merge(last_lightning, on='id_orage', suffixes=('', '_last'))
db_date['time_before_end'] = (db_date['date_last'] - db_date['date']).dt.total_seconds() / 60
db_date = db_date[['id_orage', 'lightning_id', 'time_before_end']]
db = df_ratio_fast.merge(db_date, on=['id_orage', 'lightning_id'], how='left')
db['season'] = ''
print(db.columns)
db['Date'] = pd.to_datetime(db['date'])
db.loc[db['Date'].dt.month.isin([12, 1, 2]), 'season'] = 'hiver'
db.loc[db['Date'].dt.month.isin([3, 4, 5]), 'season'] = 'printemps'
db.loc[db['Date'].dt.month.isin([6, 7, 8]), 'season'] = 'été'
db.loc[db['Date'].dt.month.isin([9, 10, 11]), 'season'] = 'automne'
db = pd.get_dummies(db, columns=['airport'], prefix='airport', dtype=int)
airport_cols = [col for col in db.columns if col.startswith('airport_')]
db['datetime'] = pd.to_datetime(db['Date'], utc=True)
db['hour'] = db['datetime'].dt.hour
db['month'] = db['datetime'].dt.month
db['dayofyear'] = db['datetime'].dt.dayofyear
db['hour_sin'] = np.sin(2 * np.pi * db['hour'] / 24)
db['hour_cos'] = np.cos(2 * np.pi * db['hour'] / 24)
db['month_sin'] = np.sin(2 * np.pi * db['month'] / 12)
db['month_cos'] = np.cos(2 * np.pi * db['month'] / 12)
db['dayofyear_sin'] = np.sin(2 * np.pi * db['dayofyear'] / 365)
db['dayofyear_cos'] = np.cos(2 * np.pi * db['dayofyear'] / 365)
db = pd.get_dummies(db, columns=['season'], prefix='season', dtype=int)
db['icloud'] = db['icloud'].astype(int)
db['datetime'] = pd.to_datetime(db['Date'], utc=True)
db = db.sort_values(['datetime', 'lightning_id']).reset_index(drop=True)
db.to_csv("db_finale.csv", index=False)