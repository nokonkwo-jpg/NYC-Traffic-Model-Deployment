# SegmentedModeling.py

import pandas as pd
import numpy as np
import glob
import holidays
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance


class SegmentedModel:
    """
    A segmented traffic-volume model that trains separate HistGradientBoostingRegressor
    instances for normal periods and event periods (holidays, heavy snow).
    """
    def __init__(
        self,
        normal_params=None,
        event_params=None,
        oversample_factor: int = 5
    ):
        # default hyperparameters if none provided
        self.normal_params = normal_params or {
            'max_iter': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'early_stopping': True,
            'random_state': 42
        }
        self.event_params = event_params or self.normal_params.copy()
        self.oversample_factor = oversample_factor
        self.model_normal = None
        self.model_event = None
        self.features = []
        self.train_df = None
        self.test_df = None

    def fit(
        self,
        df: pd.DataFrame,
        features: list,
        target_col: str = 'Vol_log',
        event_cols: tuple = ('is_holiday', 'heavy_snow'),
        train_frac: float = 0.8
    ):
        """
        Fit segmented models on df. Splits df into train/test by time index,
        then trains normal and event models on respective subsets.
        """
        # keep features and target
        self.features = features
        # drop invalid rows
        df_clean = df.dropna(subset=features + [target_col]).reset_index(drop=True)
        # time-based split
        split_idx = int(len(df_clean) * train_frac)
        self.train_df = df_clean.iloc[:split_idx].copy()
        self.test_df = df_clean.iloc[split_idx:].copy()

        # define event mask for training
        mask_event = (
            self.train_df[event_cols[0]].astype(bool)
            | self.train_df[event_cols[1]].astype(bool)
        )
        mask_normal = ~mask_event

        # train normal-period model
        self.model_normal = HistGradientBoostingRegressor(**self.normal_params)
        self.model_normal.fit(
            self.train_df.loc[mask_normal, features],
            self.train_df.loc[mask_normal, target_col]
        )

        # prepare event-period data (with optional oversampling)
        evt_df = self.train_df.loc[mask_event].copy()
        if len(evt_df) < len(self.train_df) * 0.1:
            evt_df = pd.concat(
                [evt_df] * self.oversample_factor,
                ignore_index=True
            ).sample(frac=1, random_state=42)

        # train event-period model
        self.model_event = HistGradientBoostingRegressor(**self.event_params)
        self.model_event.fit(evt_df[features], evt_df[target_col])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict raw volume (un-transformed) on df.
        """
        mask_event = (
            df['is_holiday'].astype(bool)
            | df['heavy_snow'].astype(bool)
        )
        X = df[self.features]
        # log-volume predictions
        pred_log = np.where(
            mask_event,
            self.model_event.predict(X),
            self.model_normal.predict(X)
        )
        # back-transform
        return np.expm1(pred_log)

    def evaluate(self):
        """
        Compute and print metrics for the segmented model vs. baseline, on the held-out test set.
        Also prints permutation importances for the normal-period model.
        """
        if self.test_df is None:
            raise ValueError("Must call fit before evaluate.")

        y_true = self.test_df['Vol']
        y_pred = self.predict(self.test_df)

        # mask zeros for MAPE
        nz = y_true > 0

        # segmented metrics
        print("Segmented Model Evaluation:")
        print(f"  Test R²   (raw): {r2_score(y_true, y_pred):.4f}")
        print(f"  Test MAE  (raw): {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"  Test MAPE (raw): {mean_absolute_percentage_error(y_true[nz], y_pred[nz])*100:.2f}%")

        # baseline (normal-only)
        y_base = np.expm1(
            self.model_normal.predict(self.test_df[self.features])
        )
        print("Baseline (Normal-Only) Evaluation:")
        print(f"  Test R²   (raw): {r2_score(y_true, y_base):.4f}")
        print(f"  Test MAE  (raw): {mean_absolute_error(y_true, y_base):.2f}")
        print(f"  Test MAPE (raw): {mean_absolute_percentage_error(y_true[nz], y_base[nz])*100:.2f}%")

        # permutation importances (non-event subset)
        mask_ne = ~(
            self.test_df['is_holiday'].astype(bool)
            | self.test_df['heavy_snow'].astype(bool)
        )
        perm = permutation_importance(
            self.model_normal,
            self.test_df.loc[mask_ne, self.features],
            np.log1p(self.test_df.loc[mask_ne, 'Vol']),
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        imps = perm.importances_mean
        feat_imp = sorted(
            zip(self.features, imps), key=lambda x: x[1], reverse=True
        )
        print("\nFeature importances (permutation) for normal-period:")
        for name, imp in feat_imp:
            print(f"{name}: {imp:.6f}")

    def plot(self, n_samples: int = 1000):
        """
        Plot the first n_samples of actual vs. segmented vs. baseline predictions.
        """
        if self.test_df is None:
            raise ValueError("Must call fit before plot.")
        y_true = self.test_df['Vol'].values
        y_pred = self.predict(self.test_df)
        y_base = np.expm1(
            self.model_normal.predict(self.test_df[self.features])
        )
        N = min(n_samples, len(y_true))

        plt.figure(figsize=(12, 5))
        plt.plot(y_true[:N], label='Actual', alpha=0.7)
        plt.plot(y_pred[:N], label='Segmented', alpha=0.7)
        plt.plot(y_base[:N], label='Baseline', alpha=0.7)
        plt.title(f'Actual vs Segmented vs Baseline (first {N} samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Traffic Volume')
        plt.legend()
        plt.tight_layout()
        plt.show()


def load_and_prepare_data(
    weather_pattern: str = "RawDataFiles/weather_data_vm*.csv",
    traffic_path: str = "RawDataFiles/automated_traffic_volume_counts.csv"
) -> (pd.DataFrame, list):
    """
    Load raw weather and traffic data, engineer features and event flags,
    and return a cleaned DataFrame plus the list of feature column names.
    """
    # 1) Weather data
    weather_files = sorted(glob.glob(weather_pattern))
    if not weather_files:
        raise FileNotFoundError(f"No files matching {weather_pattern}")
    weather_chunks = []
    usecols_weather = ['date','precipitation','temperature_2m','cloud_cover_low','snow_depth']
    for file in weather_files:
        for chunk in pd.read_csv(
            file,
            usecols=usecols_weather,
            parse_dates=['date'],
            chunksize=200_000
        ):
            chunk['Yr'] = chunk['date'].dt.year
            chunk['M']  = chunk['date'].dt.month
            chunk['D']  = chunk['date'].dt.day
            chunk['HH'] = chunk['date'].dt.hour
            weather_chunks.append(chunk)
    weather_df = pd.concat(weather_chunks, ignore_index=True)

    # 2) Traffic data
    df_traf = pd.read_csv(
        traffic_path,
        usecols=['Yr','M','D','HH','Boro','Vol']
    ).rename(columns={'Boro':'borough'})

    # 3) Merge & sort
    df = pd.merge(
        df_traf, weather_df,
        on=['Yr','M','D','HH'], how='inner'
    )
    df.sort_values('date', inplace=True)

    # 4) Feature engineering
    df['hour_sin']  = np.sin(2*np.pi*df['HH']/24)
    df['hour_cos']  = np.cos(2*np.pi*df['HH']/24)
    df['weekday']   = df['date'].dt.weekday
    df['wd_sin']    = np.sin(2*np.pi*df['weekday']/7)
    df['wd_cos']    = np.cos(2*np.pi*df['weekday']/7)
    df['month_sin'] = np.sin(2*np.pi*df['M']/12)
    df['month_cos'] = np.cos(2*np.pi*df['M']/12)
    df['is_holiday'] = df['date'].dt.date.isin(
        holidays.US(years=df['Yr'].unique())
    ).astype(int)
    df['heavy_snow'] = (df['snow_depth'] > 5).astype(int)
    for lag in [1]:
        df[f'vol_lag_{lag}'] = df.groupby('borough')['Vol'].shift(lag)
    for w in [3,24]:
        df[f'vol_roll_{w}h'] = (
            df.groupby('borough')['Vol']
              .rolling(window=w, min_periods=1)
              .mean()
              .shift(1)
              .reset_index(level=0, drop=True)
        )
    df['Vol_log'] = np.log1p(df['Vol'])

    features = [
        'hour_sin','hour_cos','wd_sin','wd_cos',
        'month_sin','month_cos',
        'vol_lag_1','vol_roll_3h','vol_roll_24h'
    ]
    df_clean = df.dropna(subset=features + ['Vol_log']).reset_index(drop=True)
    return df_clean, features


def main():
    # load and prepare
    df, features = load_and_prepare_data()

    # train & evaluate segmented model
    seg_model = SegmentedModel()
    seg_model.fit(df, features)
    seg_model.evaluate()
    seg_model.plot()


if __name__ == "__main__":
    main()
