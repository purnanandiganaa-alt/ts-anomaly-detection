"""
Data loading module for sensor feature dataset.

This module handles loading and preprocessing sensor feature data from CSV files.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_sensor_features(file_path: str) -> pd.DataFrame:
    """
    Load sensor features from CSV file.
    
    Args:
        file_path (str): Path to the sensor features CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset with sensor features
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sensor features file not found: {file_path}")
    
    print(f"📂 Loading sensor feature dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset shape: {df.shape}")
    
    return df


def clean_sensor_data(df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Clean sensor data by removing metadata and interpretation columns.
    
    Args:
        df (pd.DataFrame): Original sensor dataframe
        columns_to_drop (list): List of column names to remove. 
                               Defaults to metadata columns if None.
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if columns_to_drop is None:
        columns_to_drop = [
            "category",
            "final_category",
            "final_interpretation"
        ]
    
    print("🧹 Removing metadata / interpretation columns...")
    df_clean = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    print(f"✅ New shape after cleanup: {df_clean.shape}")
    
    return df_clean


def prepare_feature_matrix(df: pd.DataFrame, exclude_cols: list = None) -> tuple:
    """
    Prepare feature matrix by extracting numeric features.
    
    Args:
        df (pd.DataFrame): Cleaned sensor dataframe
        exclude_cols (list): Columns to exclude from feature matrix (e.g., 'sensor')
        
    Returns:
        tuple: (feature_matrix, sensor_names)
    """
    if exclude_cols is None:
        exclude_cols = ["sensor"]
    
    print("📊 Preparing numeric feature matrix for scoring...")
    
    feature_cols = df.columns.drop(exclude_cols)
    X = df[feature_cols].copy()
    
    print(f"✅ Feature matrix shape: {X.shape}")
    
    if "sensor" in df.columns:
        sensor_names = df["sensor"].values
        return X, sensor_names
    
    return X, None


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Dataframe with potential missing values
        strategy (str): Strategy to use - 'mean', 'median', 'drop', or 'forward_fill'
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    print(f"🔧 Handling missing values using strategy: {strategy}...")
    
    if strategy == "drop":
        df_handled = df.dropna()
    elif strategy == "mean":
        df_handled = df.fillna(df.mean())
    elif strategy == "median":
        df_handled = df.fillna(df.median())
    elif strategy == "forward_fill":
        df_handled = df.fillna(method='ffill')
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"✅ Handled missing values. New shape: {df_handled.shape}")
    
    return df_handled


def get_data_info(df: pd.DataFrame) -> None:
    """
    Print detailed information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataframe to analyze
    """
    print("\n📋 Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
