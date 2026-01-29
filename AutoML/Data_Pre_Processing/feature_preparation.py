"""
Feature Preparation Module for AIDEX
Transforms cleaned data into model-ready representations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class FeaturePreparator:
    """
    Automated feature preparation for machine learning models.
    Handles encoding, scaling, feature engineering, and feature selection.
    """
    
    def __init__(self, task_type: str = 'classification', scaling_method: str = 'standard'):
        """
        Initialize the Feature Preparator
        
        Args:
            task_type: 'classification' or 'regression'
            scaling_method: 'standard', 'minmax', 'robust', or None
        """
        self.task_type = task_type.lower()
        self.scaling_method = scaling_method
        self.encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_stats = {}
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.missing_value_strategy = None
        
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically identify numerical, categorical, and binary features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with feature types
        """
        feature_types = {
            'numerical': [],
            'categorical': [],
            'binary': [],
            'datetime': []
        }
        
        for col in df.columns:
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types['datetime'].append(col)
            # Check for numerical
            elif pd.api.types.is_numeric_dtype(df[col]):
                unique_values = df[col].nunique()
                if unique_values == 2:
                    feature_types['binary'].append(col)
                else:
                    feature_types['numerical'].append(col)
            # Categorical
            else:
                unique_values = df[col].nunique()
                if unique_values == 2:
                    feature_types['binary'].append(col)
                else:
                    feature_types['categorical'].append(col)
        
        self.feature_stats['types'] = feature_types
        return feature_types
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto', fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values using various imputation strategies
        
        Args:
            df: Input dataframe
            strategy: 'auto', 'mean', 'median', 'mode', 'constant', 'knn', or 'drop'
                     'auto' uses median for numerical and mode for categorical
            fit: Whether to fit imputers (True for training, False for inference)
            
        Returns:
            Dataframe with imputed values
        """
        df_imputed = df.copy()
        
        # Check if there are any missing values
        if df_imputed.isnull().sum().sum() == 0:
            return df_imputed
        
        if fit:
            self.missing_value_strategy = strategy
        
        feature_types = self.feature_stats.get('types', self.identify_feature_types(df))
        
        # Separate numerical and categorical columns
        numerical_cols = feature_types['numerical'] + feature_types['binary']
        categorical_cols = feature_types['categorical']
        
        # Filter to only columns that exist in the dataframe and have missing values
        numerical_cols = [col for col in numerical_cols if col in df_imputed.columns and df_imputed[col].isnull().any()]
        categorical_cols = [col for col in categorical_cols if col in df_imputed.columns and df_imputed[col].isnull().any()]
        
        # Handle strategy='drop' - remove rows with missing values
        if strategy == 'drop':
            initial_rows = len(df_imputed)
            df_imputed = df_imputed.dropna()
            dropped_rows = initial_rows - len(df_imputed)
            if dropped_rows > 0:
                print(f"   Dropped {dropped_rows} rows with missing values")
            return df_imputed
        
        # Handle numerical features
        if numerical_cols:
            if strategy == 'auto' or strategy == 'median':
                impute_strategy = 'median'
            elif strategy == 'mean':
                impute_strategy = 'mean'
            elif strategy == 'constant':
                impute_strategy = 'constant'
            elif strategy == 'knn':
                if fit:
                    self.numerical_imputer = KNNImputer(n_neighbors=5)
                    df_imputed[numerical_cols] = self.numerical_imputer.fit_transform(df_imputed[numerical_cols])
                else:
                    if self.numerical_imputer is not None:
                        df_imputed[numerical_cols] = self.numerical_imputer.transform(df_imputed[numerical_cols])
                impute_strategy = None  # Already handled
            else:
                impute_strategy = 'median'  # Default
            
            if impute_strategy:
                if fit:
                    self.numerical_imputer = SimpleImputer(strategy=impute_strategy, fill_value=0 if impute_strategy == 'constant' else None)
                    df_imputed[numerical_cols] = self.numerical_imputer.fit_transform(df_imputed[numerical_cols])
                else:
                    if self.numerical_imputer is not None:
                        df_imputed[numerical_cols] = self.numerical_imputer.transform(df_imputed[numerical_cols])
        
        # Handle categorical features
        if categorical_cols:
            if strategy == 'auto' or strategy == 'mode':
                impute_strategy = 'most_frequent'
            elif strategy == 'constant':
                impute_strategy = 'constant'
            else:
                impute_strategy = 'most_frequent'  # Default
            
            if fit:
                self.categorical_imputer = SimpleImputer(strategy=impute_strategy, fill_value='missing' if impute_strategy == 'constant' else None)
                df_imputed[categorical_cols] = self.categorical_imputer.fit_transform(df_imputed[categorical_cols].astype(str))
            else:
                if self.categorical_imputer is not None:
                    df_imputed[categorical_cols] = self.categorical_imputer.transform(df_imputed[categorical_cols].astype(str))
        
        return df_imputed
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding strategies
        
        Args:
            df: Input dataframe
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            Dataframe with encoded features
        """
        df_encoded = df.copy()
        feature_types = self.feature_stats.get('types', self.identify_feature_types(df))
        
        # Handle binary features with Label Encoding
        for col in feature_types['binary']:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        le = self.encoders[col]
                        df_encoded[col] = df_encoded[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        # Handle categorical features with One-Hot Encoding (for low cardinality)
        # or Label Encoding (for high cardinality)
        for col in feature_types['categorical']:
            if col in df_encoded.columns:
                cardinality = df_encoded[col].nunique()
                
                if cardinality <= 10:  # One-Hot Encoding for low cardinality
                    if fit:
                        # Get dummies and store column names
                        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                        self.encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                        df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                    else:
                        if col in self.encoders and self.encoders[col]['type'] == 'onehot':
                            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                            # Align with training columns
                            for dummy_col in self.encoders[col]['columns']:
                                if dummy_col not in dummies.columns:
                                    dummies[dummy_col] = 0
                            dummies = dummies[self.encoders[col]['columns']]
                            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                else:  # Label Encoding for high cardinality
                    if fit:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        self.encoders[col] = {'type': 'label', 'encoder': le}
                    else:
                        if col in self.encoders and self.encoders[col]['type'] == 'label':
                            le = self.encoders[col]['encoder']
                            df_encoded[col] = df_encoded[col].astype(str).map(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
        
        return df_encoded
    
    def engineer_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract useful features from datetime columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered datetime features
        """
        df_engineered = df.copy()
        feature_types = self.feature_stats.get('types', self.identify_feature_types(df))
        
        for col in feature_types['datetime']:
            if col in df_engineered.columns:
                # Extract components
                df_engineered[f'{col}_year'] = df_engineered[col].dt.year
                df_engineered[f'{col}_month'] = df_engineered[col].dt.month
                df_engineered[f'{col}_day'] = df_engineered[col].dt.day
                df_engineered[f'{col}_dayofweek'] = df_engineered[col].dt.dayofweek
                df_engineered[f'{col}_quarter'] = df_engineered[col].dt.quarter
                df_engineered[f'{col}_is_weekend'] = df_engineered[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Drop original datetime column
                df_engineered = df_engineered.drop(col, axis=1)
        
        return df_engineered
    
    def create_interaction_features(self, df: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """
        Create polynomial and interaction features for numerical columns
        
        Args:
            df: Input dataframe
            max_interactions: Maximum number of interaction features to create
            
        Returns:
            Dataframe with interaction features
        """
        df_interact = df.copy()
        feature_types = self.feature_stats.get('types', self.identify_feature_types(df))
        numerical_cols = feature_types['numerical']
        
        # Limit to most important numerical features to avoid explosion
        if len(numerical_cols) > 5:
            numerical_cols = numerical_cols[:5]
        
        interaction_count = 0
        for i, col1 in enumerate(numerical_cols):
            if col1 in df_interact.columns:
                # Create squared features
                df_interact[f'{col1}_squared'] = df_interact[col1] ** 2
                interaction_count += 1
                
                # Create interaction features with other columns
                for col2 in numerical_cols[i+1:]:
                    if col2 in df_interact.columns and interaction_count < max_interactions:
                        df_interact[f'{col1}_x_{col2}'] = df_interact[col1] * df_interact[col2]
                        interaction_count += 1
                        
                if interaction_count >= max_interactions:
                    break
        
        return df_interact
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using the specified scaling method
        
        Args:
            df: Input dataframe
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Dataframe with scaled features
        """
        if self.scaling_method is None:
            return df
        
        df_scaled = df.copy()
        
        # Identify numerical columns (all columns should be numerical at this point)
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) == 0:
            return df_scaled
        
        if fit:
            # Initialize scaler based on method
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                return df_scaled
            
            df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
        else:
            if self.scaler is not None:
                df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        return df_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', 
                       k: int = 'all') -> pd.DataFrame:
        """
        Select the most important features
        
        Args:
            X: Feature dataframe
            y: Target variable
            method: 'mutual_info', 'f_test', or 'all'
            k: Number of top features to select, or 'all'
            
        Returns:
            Dataframe with selected features
        """
        if k == 'all' or k >= X.shape[1]:
            self.selected_features = X.columns.tolist()
            return X
        
        # Choose scoring function based on task type and method
        if method == 'mutual_info':
            if self.task_type == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        else:  # f_test
            if self.task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        
        # Perform feature selection
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = None, 
                        fit: bool = True, create_interactions: bool = False,
                        select_features: bool = False, k_features: int = 'all',
                        handle_missing: str = 'auto') -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Complete feature preparation pipeline
        
        Args:
            df: Input dataframe
            target_col: Name of target column (if present in df)
            fit: Whether to fit transformers (True for training, False for inference)
            create_interactions: Whether to create interaction features
            select_features: Whether to perform feature selection
            k_features: Number of features to select (if select_features=True)
            handle_missing: Strategy for missing values ('auto', 'mean', 'median', 'mode', 'knn', 'drop', or None)
            
        Returns:
            Tuple of (prepared features dataframe, target series if applicable)
        """
        # Separate target if present
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(target_col, axis=1)
        
        # Step 1: Identify feature types
        if fit:
            self.identify_feature_types(df)
        
        # Step 2: Handle missing values (NEW!)
        if handle_missing:
            df = self.handle_missing_values(df, strategy=handle_missing, fit=fit)
        
        # Step 3: Engineer datetime features
        df = self.engineer_datetime_features(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Step 5: Create interaction features (optional)
        if create_interactions and fit:
            df = self.create_interaction_features(df)
        
        # Step 6: Scale features
        df = self.scale_features(df, fit=fit)
        
        # Step 7: Feature selection (optional)
        if select_features and y is not None and fit:
            df = self.select_features(df, y, k=k_features)
        elif select_features and not fit and self.selected_features:
            # Use previously selected features
            df = df[self.selected_features]
        
        return df, y
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the feature preparation process
        
        Returns:
            Dictionary with feature statistics and transformation info
        """
        info = {
            'feature_types': self.feature_stats.get('types', {}),
            'scaling_method': self.scaling_method,
            'num_encoders': len(self.encoders),
            'selected_features': self.selected_features,
            'num_final_features': len(self.selected_features) if self.selected_features else None
        }
        return info


# Utility functions
def auto_prepare_features(df: pd.DataFrame, target_col: str, task_type: str = 'classification',
                         scaling_method: str = 'standard', create_interactions: bool = False,
                         select_features: bool = False, k_features: int = 'all',
                         handle_missing: str = 'auto') -> Tuple[pd.DataFrame, pd.Series, FeaturePreparator]:
    """
    Convenience function for automatic feature preparation
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        task_type: 'classification' or 'regression'
        scaling_method: 'standard', 'minmax', 'robust', or None
        create_interactions: Whether to create interaction features
        select_features: Whether to perform feature selection
        k_features: Number of features to select
        handle_missing: Strategy for missing values ('auto', 'mean', 'median', 'mode', 'knn', 'drop', or None)
        
    Returns:
        Tuple of (prepared features, target, fitted preparator)
    """
    preparator = FeaturePreparator(task_type=task_type, scaling_method=scaling_method)
    X, y = preparator.prepare_features(
        df, 
        target_col=target_col, 
        fit=True,
        create_interactions=create_interactions,
        select_features=select_features,
        k_features=k_features,
        handle_missing=handle_missing
    )
    
    return X, y, preparator


if __name__ == "__main__":
    # Example usage
    print("Feature Preparation Module for AIDEX")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [50000, 60000, 75000, 90000, 120000],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'purchased': [0, 1, 1, 0, 1]
    })
    
    print("\nOriginal Data:")
    print(sample_data)
    
    # Prepare features
    X, y, preparator = auto_prepare_features(
        sample_data, 
        target_col='purchased',
        task_type='classification',
        scaling_method='standard'
    )
    
    print("\nPrepared Features:")
    print(X)
    print("\nTarget:")
    print(y)
    print("\nFeature Info:")
    print(preparator.get_feature_info())
