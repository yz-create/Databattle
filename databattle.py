from calendar import month

import pandas as pd
import numpy as np
import datetime as dt
print(db[["id_orage"]].nunique())
print(db[["id_orage"]].head(100))
###Modification de la bdd
db = pd.read_csv("df_add_ratio_nb_eclairs_ect.csv")
db_date = db[['id_orage', 'date', 'lightning_id']]
db_date['time_before_end'] = np.nan
db_date['date'] = pd.to_datetime(db_date['date'])
db_date = db_date.sort_values(by=['id_orage', 'date'])
last_lightning = db_date.groupby('id_orage')['date'].max()
db_date = db_date.merge(last_lightning, on='id_orage', suffixes=('', '_last'))
db_date['time_before_end'] = (db_date['date_last'] - db_date['date']).dt.total_seconds() / 60
db_date = db_date[['id_orage', 'lightning_id', 'time_before_end']]
db = db.merge(db_date, on=['id_orage', 'lightning_id'], how='left')
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

features = [
    'lon', 'lat', 'amplitude', 'maxis', 'icloud', 'dist', 'azimuth', 'ic',
    'ratio_ic_0_5', 'ratio_ic_5_10', 'ratio_ic_10_20', 'ratio_ic_20_120',
    'n_0_5_0_20', 'n_5_10_0_20', 'n_10_20_0_20', 'n_20_120_0_20',
    'n_0_5_20_30', 'n_5_10_20_30', 'n_10_20_20_30', 'n_20_120_20_30',
    'hour_cos', 'hour_sin', 'month_cos','month_sin', 'dayofyear_cos', 'dayofyear_sin', 'season_automne', 'season_hiver', 'season_printemps', 'season_été'
] + airport_cols

print(db.columns)
series_ids = sorted(db["id_orage"].unique())
split_idx = int(len(series_ids) * 0.8)

train_series = series_ids[:split_idx]
test_series = series_ids[split_idx:]

print("=" * 80)
print("SPLIT PAR SERIES")
print(f"Nombre total de séries : {len(series_ids)}")
print(f"Train séries           : {len(train_series)}")
print(f"Test séries            : {len(test_series)}")
print(f"Premières train series : {train_series[:5]}")
print(f"Dernières train series : {train_series[-5:] if len(train_series) >= 5 else train_series}")
print(f"Premières test series  : {test_series[:5]}")
print(f"Dernières test series  : {test_series[-5:] if len(test_series) >= 5 else test_series}")
print("=" * 80)

train_df = db[db["id_orage"].isin(train_series)].copy()
test_df = db[db["id_orage"].isin(test_series)].copy()

print("\nTAILLE DES DATAFRAMES")
print(f"train_df shape         : {train_df.shape}")
print(f"test_df shape          : {test_df.shape}")

print("\nFEATURES")
print(features)

print("\nAPERCU TRAIN")
print(train_df[["id_orage", "datetime", "time_before_end"] + features].head())

print("\nAPERCU TEST")
print(test_df[["id_orage", "datetime", "time_before_end"] + features].head())

X_train = train_df[features]
y_train = train_df["target_log"]

X_test = test_df[features]
y_test_true = test_df["time_before_end"]

print("\nSTATS TARGET TRAIN (échelle réelle)")
print(np.expm1(y_train).describe())

print("\nSTATS TARGET TEST (échelle réelle)")
print(y_test_true.describe())

print("\nVALEURS MANQUANTES")
print("NaN X_train :", X_train.isna().sum().sum())
print("NaN X_test  :", X_test.isna().sum().sum())
print("NaN y_train :", y_train.isna().sum())
print("NaN y_test  :", y_test_true.isna().sum())

model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("\n" + "=" * 80)
print("FIT...")
model.fit(X_train, y_train)
print("FIT TERMINE")
print("=" * 80)

print("\nPREDICT...")
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
print("PREDICT TERMINE")

results = test_df[["id_orage", "datetime", "time_before_end"]].copy()
results = results.rename(columns={"time_before_end": "y_true"})
results["y_pred"] = y_pred
results["abs_err"] = (results["y_true"] - results["y_pred"]).abs()
results["err"] = results["y_pred"] - results["y_true"]

print("\nHEAD RESULTS")
print(results.head(10))

print("\nTAIL RESULTS")
print(results.tail(10))

print("\nSTATS PREDICTIONS")
print(results[["y_true", "y_pred", "abs_err", "err"]].describe())

mae = results["abs_err"].mean()
rmse = np.sqrt(((results["y_true"] - results["y_pred"]) ** 2).mean())

print("\nMETRIQUES")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")

print("\nQUELQUES EXEMPLES")
for idx in results.index[:10]:
    row = results.loc[idx]
    print(
        f"id_orage={row['id_orage']} | "
        f"datetime={row['datetime']} | "
        f"y_true={row['y_true']:.6f} | "
        f"y_pred={row['y_pred']:.6f} | "
        f"abs_err={row['abs_err']:.6f}"
    )

print("\nIMPORTANCES")
for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{f:15s} -> {imp:.6f}")
