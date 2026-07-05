"""
Feature/sensor selection module for anomaly detection pipeline.

This module implements sensor selection based on statistical feature scoring
and correlation analysis to identify the most useful sensors for anomaly detection.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class SensorSelector:
    """
    Sensor selection engine based on feature correlation and weighted scoring.
    """
    
    def __init__(self, correlation_threshold: float = 0.85):
        """
        Initialize the SensorSelector.
        
        Args:
            correlation_threshold (float): Threshold for removing highly correlated features
        """
        self.correlation_threshold = correlation_threshold
        self.scaler = MinMaxScaler()
        self.feature_weights = {
            "mean": 0.25,
            "skew": 0.15,
            "num_peaks": 0.20,
            "nunique": 0.10,
            "trend_strength": 0.15,
            "sensor_score": 0.15
        }
        self.scaled_features = None
        
    def compute_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute absolute correlation matrix.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: Absolute correlation matrix
        """
        print("📊 Computing correlation matrix...")
        corr_matrix = X.corr().abs()
        return corr_matrix
    
    def remove_correlated_features(self, X: pd.DataFrame) -> tuple:
        """
        Remove highly correlated features above threshold.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            tuple: (filtered_features, removed_features_list)
        """
        corr_matrix = self.compute_correlation_matrix(X)
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        
        print(f"🔴 Highly correlated features to remove (threshold={self.correlation_threshold}):")
        print(to_drop)
        
        X_filtered = X.drop(columns=to_drop)
        print(f"✅ Shape after correlation filtering: {X_filtered.shape}")
        
        return X_filtered, to_drop
    
    def normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using MinMaxScaler.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: Normalized feature matrix
        """
        print("📈 Normalizing feature space...")
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        print("✅ Normalization complete")
        self.scaled_features = X_scaled
        
        return X_scaled
    
    def compute_sensor_scores(self, X_scaled: pd.DataFrame) -> pd.Series:
        """
        Compute final sensor scores using weighted combination of features.
        
        Args:
            X_scaled (pd.DataFrame): Normalized feature matrix
            
        Returns:
            pd.Series: Final sensor scores
        """
        print("🎯 Computing final sensor scores with weighted features...")
        
        final_score = pd.Series(np.zeros(len(X_scaled)), index=X_scaled.index)
        
        for feature, weight in self.feature_weights.items():
            if feature in X_scaled.columns:
                final_score += weight * X_scaled[feature]
                print(f"  - {feature}: {weight:.2f}")
        
        print("✅ Final scores computed")
        return final_score
    
    def select_sensors(self, X_scaled: pd.DataFrame, threshold: float = 0.45) -> pd.DataFrame:
        """
        Select sensors based on final score threshold.
        
        Args:
            X_scaled (pd.DataFrame): Normalized feature matrix
            threshold (float): Score threshold for sensor selection
            
        Returns:
            pd.DataFrame: Ranked dataframe with selected sensors
        """
        print(f"\n🎯 Selecting sensors with score threshold > {threshold}...")
        
        final_score = self.compute_sensor_scores(X_scaled)
        X_scaled["final_score"] = final_score
        
        # Rank sensors
        ranked = X_scaled.sort_values("final_score", ascending=False)
        
        # Select sensors above threshold
        selected = ranked[ranked["final_score"] > threshold]
        
        print(f"\nSelected sensors:")
        print(selected[["final_score"]])
        print(f"\nTotal selected sensors: {len(selected)}")
        
        return ranked, selected
    
    def extract_sensor_names(self, df_original: pd.DataFrame, selected_indices) -> list:
        """
        Extract sensor names from selected indices.
        
        Args:
            df_original (pd.DataFrame): Original dataframe with sensor names
            selected_indices: Index of selected sensors
            
        Returns:
            list: List of selected sensor names
        """
        if "sensor" in df_original.columns:
            selected_sensor_names = df_original.loc[selected_indices, "sensor"].tolist()
            return selected_sensor_names
        return None
    
    def save_selected_sensors(self, sensor_list: list, output_path: str) -> None:
        """
        Save selected sensor list to JSON file.
        
        Args:
            sensor_list (list): List of selected sensor names
            output_path (str): Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(sensor_list, f, indent=4)
        
        print(f"✅ Selected sensors saved to {output_path}")


def run_sensor_selection(data_path: str, output_path: str, 
                         threshold: float = 0.45,
                         correlation_threshold: float = 0.85) -> dict:
    """
    Run complete sensor selection pipeline.
    
    Args:
        data_path (str): Path to sensor features CSV file
        output_path (str): Path to save selected sensors JSON
        threshold (float): Score threshold for sensor selection
        correlation_threshold (float): Correlation threshold for feature removal
        
    Returns:
        dict: Dictionary containing results and metadata
    """
    from src.core.data_loader import load_sensor_features, clean_sensor_data, prepare_feature_matrix
    
    # Load and prepare data
    df = load_sensor_features(data_path)
    df_clean = clean_sensor_data(df)
    X, sensor_names = prepare_feature_matrix(df_clean)
    
    # Initialize selector
    selector = SensorSelector(correlation_threshold=correlation_threshold)
    
    # Remove correlated features
    X_filtered, removed_features = selector.remove_correlated_features(X)
    
    # Normalize features
    X_scaled = selector.normalize_features(X_filtered)
    
    # Select sensors
    ranked, selected = selector.select_sensors(X_scaled, threshold=threshold)
    
    # Extract and save sensor names
    selected_sensor_names = df_clean.loc[selected.index, "sensor"].tolist()
    selector.save_selected_sensors(selected_sensor_names, output_path)
    
    results = {
        "total_sensors": len(df_clean),
        "selected_sensors": len(selected),
        "selected_sensor_names": selected_sensor_names,
        "threshold": threshold,
        "removed_correlated_features": removed_features,
        "selection_scores": selected["final_score"].to_dict()
    }
    
    return results


if __name__ == "__main__":
    # Run from the repo root: python -m src.core.feature_selection
    data_path = "data/processed/df_sensor_features.csv"
    output_path = "data/processed/selected_sensors.json"
    
    results = run_sensor_selection(data_path, output_path)
    
    print("\n" + "="*50)
    print("SENSOR SELECTION SUMMARY")
    print("="*50)
    print(f"Total sensors: {results['total_sensors']}")
    print(f"Selected sensors: {results['selected_sensors']}")
    print(f"Selection threshold: {results['threshold']}")
    print(f"\nSelected sensor names: {results['selected_sensor_names']}")
