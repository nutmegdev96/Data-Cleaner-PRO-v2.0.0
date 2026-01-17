"""
Utility functions for Data Cleaner Pro
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import re
import json
from datetime import datetime
import hashlib
import warnings

warnings.filterwarnings('ignore')


def detect_file_encoding(filepath: str, sample_size: int = 1024) -> str:
    """
    Detect file encoding
    
    Parameters:
    -----------
    filepath : str
        Path to file
    sample_size : int
        Number of bytes to sample
        
    Returns:
    --------
    str
        Detected encoding
        
    #hint: Useful for reading CSV files with unusual encodings
    """
    import chardet
    
    with open(filepath, 'rb') as f:
        raw_data = f.read(sample_size)
    
    result = chardet.detect(raw_data)
    return result['encoding']


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate memory usage of DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns:
    --------
    dict
        Memory usage statistics
        
    #hint: Use this to optimize data types and reduce memory footprint
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_mb': total_memory / 1024**2,
        'total_kb': total_memory / 1024,
        'per_column_mb': (memory_usage / 1024**2).to_dict(),
        'dtype_breakdown': df.dtypes.value_counts().to_dict()
    }


def generate_data_hash(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None) -> str:
    """
    Generate hash of data for change detection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to include in hash
        
    Returns:
    --------
    str
        MD5 hash of data
        
    #hint: Use to detect if data has changed between runs
    """
    if columns is None:
        columns = df.columns
    
    # Convert selected columns to string and concatenate
    data_string = ''
    for col in columns:
        if col in df.columns:
            data_string += df[col].astype(str).sum()
    
    return hashlib.md5(data_string.encode()).hexdigest()


def split_train_test_by_date(df: pd.DataFrame,
                           date_column: str,
                           test_size: float = 0.2,
                           gap_days: int = 0) -> tuple:
    """
    Split data into train/test sets by date
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with datetime column
    date_column : str
        Name of datetime column
    test_size : float
        Proportion of latest data for test set
    gap_days : int
        Gap between train and test sets
        
    Returns:
    --------
    tuple
        (train_df, test_df)
        
    #hint: Use for time series cross-validation
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")
    
    df = df.copy()
    df = df.sort_values(date_column)
    
    split_index = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index + gap_days:]
    
    return train_df, test_df


def calculate_data_quality_score(df: pd.DataFrame,
                                weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate overall data quality score
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    weights : dict, optional
        Weights for different quality metrics
        
    Returns:
    --------
    float
        Data quality score (0-100)
        
    #hint: Use to monitor data quality over time
    """
    if weights is None:
        weights = {
            'completeness': 0.3,
            'consistency': 0.3,
            'accuracy': 0.2,
            'timeliness': 0.1,
            'uniqueness': 0.1
        }
    
    scores = {}
    
    # Completeness score (non-null percentage)
    completeness = 1 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]))
    scores['completeness'] = completeness * 100
    
    # Consistency score (data type consistency)
    type_consistency = sum(df[col].apply(type).nunique() == 1 
                          for col in df.columns) / len(df.columns)
    scores['consistency'] = type_consistency * 100
    
    # Uniqueness score (duplicate rows)
    uniqueness = 1 - (df.duplicated().sum() / len(df))
    scores['uniqueness'] = uniqueness * 100
    
    # Calculate weighted score
    total_score = sum(scores.get(k, 0) * weights.get(k, 0) 
                     for k in weights.keys())
    
    return round(total_score, 2)


def save_cleaning_pipeline(pipeline: Any, filepath: str) -> None:
    """
    Save cleaning pipeline configuration
    
    Parameters:
    -----------
    pipeline : Any
        Pipeline object or configuration
    filepath : str
        Path to save pipeline
        
    #hint: Use for reproducible data cleaning workflows
    """
    import pickle
    
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)


def load_cleaning_pipeline(filepath: str) -> Any:
    """
    Load saved cleaning pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to saved pipeline
        
    Returns:
    --------
    Any
        Loaded pipeline
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_data_dictionary(df: pd.DataFrame,
                          descriptions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Create data dictionary for documentation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    descriptions : dict, optional
        Column descriptions
        
    Returns:
    --------
    pandas.DataFrame
        Data dictionary
        
    #hint: Essential for data governance and documentation
    """
    dict_data = []
    
    for col in df.columns:
        col_info = {
            'column_name': col,
            'data_type': str(df[col].dtype),
            'non_null_count': df[col].notna().sum(),
            'null_count': df[col].isna().sum(),
            'unique_values': df[col].nunique(),
            'sample_values': str(df[col].dropna().head(3).tolist()),
            'description': descriptions.get(col, '') if descriptions else ''
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            })
        
        dict_data.append(col_info)
    
    return pd.DataFrame(dict_data)


def export_to_sql(df: pd.DataFrame, table_name: str,
                 connection_string: str, if_exists: str = 'replace') -> None:
    """
    Export DataFrame to SQL database
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    table_name : str
        Target table name
    connection_string : str
        SQLAlchemy connection string
    if_exists : str
        What to do if table exists: 'fail', 'replace', 'append'
        
    #hint: Use for automated data pipeline to databases
    """
    from sqlalchemy import create_engine
    
    engine = create_engine(connection_string)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    
    print(f"✅ Data exported to {table_name} in database")
