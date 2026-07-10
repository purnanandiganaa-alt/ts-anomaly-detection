import pandas as pd
import numpy as np
import json
import os

def create_features(train_path, selected_sensors_path):
    print("USING UPDATED FEATURE PIPELINE")

    # ----------------------------
    # Load selected sensors
    # ----------------------------
    with open(selected_sensors_path, "r") as f:
        selected_sensors = json.load(f)

    # ----------------------------
    # Process each series separately
    # ----------------------------
    all_features = []

    files = sorted(
        [f for f in os.listdir(train_path) if f.endswith(".csv")]
    )

    for file_name in files:

        df = pd.read_csv(
            os.path.join(train_path, file_name)
        )

        # Keep only selected sensors
        df = df[selected_sensors].copy()

        # Base feature matrix
        X = df.copy()

        # ----------------------------
        # Lag features
        # ----------------------------
        for col in selected_sensors:
            X[f"{col}_lag1"] = X[col].shift(1)

        # ----------------------------
        # Delta features
        # ----------------------------
        for col in selected_sensors:
            X[f"{col}_delta"] = X[col] - X[col].shift(1)

        # ----------------------------
        # Rolling mean
        # ----------------------------
        for col in selected_sensors:
            X[f"{col}_roll_mean"] = (
                X[col]
                .rolling(window=5, min_periods=1)
                .mean()
            )

        # ----------------------------
        # Rolling std
        # ----------------------------
        for col in selected_sensors:
            X[f"{col}_roll_std"] = (
                X[col]
                .rolling(window=5, min_periods=1)
                .std()
            )

        # ----------------------------
        # EWMA
        # ----------------------------
        for col in selected_sensors:
            X[f"{col}_ewma"] = (
                X[col]
                .ewm(span=5)
                .mean()
            )

        # ----------------------------
        # Interaction features
        # ----------------------------
        interaction_features = {}

        sensor_list = list(selected_sensors)

        for i in range(len(sensor_list)):
            for j in range(i + 1, len(sensor_list)):
                s1 = sensor_list[i]
                s2 = sensor_list[j]

                interaction_features[
                    f"{s1}_x_{s2}"
                ] = X[s1] * X[s2]

        X = pd.concat(
            [X, pd.DataFrame(interaction_features)],
            axis=1
        )

        # ----------------------------
        # Clean NaNs INSIDE SERIES
        # ----------------------------
        X = (
            X
            .bfill()
            .ffill()
            .fillna(0)
        )

        all_features.append(X)

    # ----------------------------
    # Concatenate processed series
    # ----------------------------
    X_clean = pd.concat(
        all_features,
        ignore_index=True
    )

    return X_clean