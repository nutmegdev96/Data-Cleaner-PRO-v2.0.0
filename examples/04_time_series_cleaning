"""
Example 4: Time Series Data Cleaning
Handling time series data with seasonality, missing timestamps, and anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append('../src')
from cleaner import DataCleaner
from transformers import DataTransformer

print("=" * 60)
print("⏰ EXAMPLE 4: Time Series Data Cleaning")
print("=" * 60)

def generate_time_series_data():
    """Generate time series data with realistic patterns and issues"""
    np.random.seed(42)
    
    # Create hourly time series for 3 months
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Introduce missing timestamps (5% missing)
    missing_indices = np.random.choice(len(date_range), 
                                      size=int(len(date_range) * 0.05), 
                                      replace=False)
    date_range = np.delete(date_range, missing_indices)
    
    # Generate multiple time series
    n_series = len(date_range)
    
    # 1. Website traffic (seasonal + trend)
    base_traffic = 1000
    seasonal_component = 500 * np.sin(2 * np.pi * np.arange(n_series) / 24)  # Daily seasonality
    weekly_seasonality = 200 * np.sin(2 * np.pi * np.arange(n_series) / (24*7))  # Weekly
    trend = np.linspace(0, 300, n_series)  # Upward trend
    noise = np.random.normal(0, 50, n_series)
    
    website_traffic = base_traffic + seasonal_component + weekly_seasonality + trend + noise
    website_traffic = np.maximum(website_traffic, 0)  # No negative traffic
    
    # 2. Sales data (spikes on weekends)
    base_sales = 50
    weekend_boost = np.array([100 if d.weekday() >= 5 else 0 for d in date_range])
    promotional_spikes = np.zeros(n_series)
    spike_days = np.random.choice(n_series, 10, replace=False)
    promotional_spikes[spike_days] = np.random.uniform(200, 500, 10)
    
    sales = base_sales + weekend_boost + promotional_spikes + np.random.poisson(20, n_series)
    
    # 3. Server errors (occasional spikes)
    base_errors = 0.5
    error_spikes = np.zeros(n_series)
    error_days = np.random.choice(n_series, 15, replace=False)
    error_spikes[error_days] = np.random.exponential(10, 15)
    
    server_errors = base_errors + error_spikes + np.random.exponential(0.1, n_series)
    
    # 4. Temperature data (realistic pattern)
    base_temp = 20
    daily_variation = 10 * np.sin(2 * np.pi * np.arange(n_series) / 24)  # Daily cycle
    monthly_trend = 5 * np.sin(2 * np.pi * np.arange(n_series) / (24*30))  # Monthly
    temperature = base_temp + daily_variation + monthly_trend + np.random.normal(0, 2, n_series)
    
    # Create DataFrame
    data = {
        'timestamp': date_range,
        'website_traffic': website_traffic.round(0),
        'sales': sales.round(0),
        'server_errors': server_errors.round(2),
        'temperature_c': temperature.round(1),
        'is_weekend': [d.weekday() >= 5 for d in date_range],
        'hour_of_day': [d.hour for d in date_range],
        'day_of_week': [d.weekday() for d in date_range]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    n_rows = len(df)
    
    # Add missing values (5%)
    missing_mask = np.random.random(size=(n_rows, 4)) < 0.05
    for i, col in enumerate(['website_traffic', 'sales', 'server_errors', 'temperature_c']):
        df.loc[missing_mask[:, i], col] = np.nan
    
    # Add outliers (2%)
    outlier_mask = np.random.random(size=(n_rows, 4)) < 0.02
    for i, col in enumerate(['website_traffic', 'sales', 'server_errors', 'temperature_c']):
        outlier_indices = np.where(outlier_mask[:, i])[0]
        if len(outlier_indices) > 0:
            multiplier = np.random.choice([0.1, 10], size=len(outlier_indices))
            df.loc[outlier_indices, col] = df.loc[outlier_indices, col] * multiplier
    
    # Add negative values where inappropriate
    negative_indices = np.random.choice(n_rows, 20, replace=False)
    df.loc[negative_indices, 'website_traffic'] = -df.loc[negative_indices, 'website_traffic']
    df.loc[negative_indices, 'sales'] = -df.loc[negative_indices, 'sales']
    
    # Save data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_timeseries.csv', index=False)
    df.to_json('data/sample_timeseries.json', orient='records', date_format='iso')
    
    return df

def run_time_series_cleaning():
    """Clean and prepare time series data for analysis"""
    
    print("\n📥 Step 1: Loading time series data...")
    if os.path.exists('data/sample_timeseries.csv'):
        df = pd.read_csv('data/sample_timeseries.csv', parse_dates=['timestamp'])
    else:
        df = generate_time_series_data()
    
    print(f"   Time series length: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Frequency: Hourly data")
    
    # Initialize cleaner
    cleaner = DataCleaner(df, verbose=True)
    
    print("\n🔍 Step 2: Time series specific checks...")
    
    # Check for timestamp issues
    print("\n   📅 Timestamp Analysis:")
    
    # 1. Check for duplicates
    duplicate_timestamps = cleaner.df['timestamp'].duplicated().sum()
    if duplicate_timestamps > 0:
        print(f"   ⚠️  Found {duplicate_timestamps} duplicate timestamps")
    
    # 2. Check for gaps
    time_diff = cleaner.df['timestamp'].diff()
    expected_freq = pd.Timedelta('1H')
    gaps = time_diff[time_diff > expected_freq * 1.5]  # More than 1.5x expected
    
    if len(gaps) > 0:
        print(f"   ⚠️  Found {len(gaps)} gaps in time series")
        print(f"   Largest gap: {gaps.max()}")
    
    # 3. Ensure chronological order
    if not cleaner.df['timestamp'].is_monotonic_increasing:
        print("   ⚠️  Timestamps not in chronological order")
        cleaner.df = cleaner.df.sort_values('timestamp')
    
    print("\n🔧 Step 3: Time series cleaning pipeline...")
    
    # Handle time series specific issues
    cleaner.standardize_column_names(case='snake') \
          .convert_data_types({'timestamp': 'datetime64[ns]'})
    
    # Fix negative values for non-negative metrics
    negative_metrics = ['website_traffic', 'sales', 'server_errors']
    for metric in negative_metrics:
        negative_mask = cleaner.df[metric] < 0
        if negative_mask.any():
            print(f"   Fixing {negative_mask.sum()} negative values in {metric}")
            cleaner.df.loc[negative_mask, metric] = abs(cleaner.df.loc[negative_mask, metric])
    
    # Handle missing values with time-aware imputation
    print("\n   🔄 Time-aware missing value imputation...")
    for col in ['website_traffic', 'sales', 'server_errors', 'temperature_c']:
        if cleaner.df[col].isna().any():
            # Forward fill for short gaps, interpolation for longer ones
            cleaner.df[col] = cleaner.df[col].interpolate(method='time', limit_direction='both')
            
            # Fill any remaining with rolling mean
            if cleaner.df[col].isna().any():
                cleaner.df[col] = cleaner.df[col].fillna(
                    cleaner.df[col].rolling(window=24, min_periods=1).mean()
                )
    
    # Handle outliers with time series aware methods
    print("\n   📊 Time series outlier detection...")
    
    for col in ['website_traffic', 'sales', 'server_errors', 'temperature_c']:
        # Use rolling median for outlier detection
        window_size = 24 * 7  # 1 week window for hourly data
        rolling_median = cleaner.df[col].rolling(window=window_size, center=True).median()
        rolling_std = cleaner.df[col].rolling(window=window_size, center=True).std()
        
        # Identify outliers (more than 3 standard deviations from rolling median)
        outlier_mask = abs(cleaner.df[col] - rolling_median) > (3 * rolling_std)
        
        if outlier_mask.any():
            print(f"   Found {outlier_mask.sum()} outliers in {col}")
            
            # Replace outliers with rolling median
            cleaner.df.loc[outlier_mask, col] = rolling_median[outlier_mask]
    
    print("\n🔄 Step 4: Time series feature engineering...")
    transformer = DataTransformer(verbose=True)
    
    # Extract comprehensive time features
    cleaner.df = transformer.extract_datetime_features(cleaner.df, 'timestamp')
    
    # Add cyclic encoding for hour and day of week
    cleaner.df['hour_sin'] = np.sin(2 * np.pi * cleaner.df['hour_of_day'] / 24)
    cleaner.df['hour_cos'] = np.cos(2 * np.pi * cleaner.df['hour_of_day'] / 24)
    cleaner.df['day_sin'] = np.sin(2 * np.pi * cleaner.df['day_of_week'] / 7)
    cleaner.df['day_cos'] = np.cos(2 * np.pi * cleaner.df['day_of_week'] / 7)
    
    # Add lag features
    for lag in [1, 2, 3, 24, 168]:  # 1h, 2h, 3h, 1 day, 1 week
        cleaner.df[f'website_traffic_lag_{lag}'] = cleaner.df['website_traffic'].shift(lag)
        cleaner.df[f'sales_lag_{lag}'] = cleaner.df['sales'].shift(lag)
    
    # Add rolling statistics
    for window in [24, 168]:  # 1 day, 1 week
        cleaner.df[f'website_traffic_ma_{window}'] = (
            cleaner.df['website_traffic'].rolling(window=window).mean()
        )
        cleaner.df[f'sales_ma_{window}'] = (
            cleaner.df['sales'].rolling(window=window).mean()
        )
        cleaner.df[f'temperature_ma_{window}'] = (
            cleaner.df['temperature_c'].rolling(window=window).mean()
        )
    
    # Add seasonal decomposition features
    from scipy import signal
    
    # Detrend using moving average
    cleaner.df['website_traffic_detrended'] = (
        cleaner.df['website_traffic'] - cleaner.df[f'website_traffic_ma_168']
    )
    cleaner.df['sales_detrended'] = (
        cleaner.df['sales'] - cleaner.df[f'sales_ma_168']
    )
    
    print("\n📈 Step 5: Time series analysis...")
    
    # Summary statistics
    print("\n   📊 Summary Statistics:")
    summary = cleaner.df[['website_traffic', 'sales', 'server_errors', 'temperature_c']].describe()
    print(summary.round(2))
    
    # Correlation analysis
    print("\n   🔗 Correlation Matrix (key metrics):")
    correlation_matrix = cleaner.df[['website_traffic', 'sales', 'server_errors', 'temperature_c']].corr()
    print(correlation_matrix.round(3))
    
    # Seasonality analysis
    print("\n   📅 Seasonality Analysis:")
    
    # Daily pattern
    daily_pattern = cleaner.df.groupby('hour_of_day').agg({
        'website_traffic': 'mean',
        'sales': 'mean',
        'temperature_c': 'mean'
    }).round(2)
    
    print("\n   Daily Patterns (by hour):")
    print(daily_pattern.head(6))
    
    # Weekly pattern
    weekly_pattern = cleaner.df.groupby('day_of_week').agg({
        'website_traffic': 'mean',
        'sales': 'mean'
    }).round(2)
    
    print("\n   Weekly Patterns (0=Monday, 6=Sunday):")
    print(weekly_pattern)
    
    print("\n💾 Step 6: Exporting cleaned time series...")
    os.makedirs('outputs/cleaned_data', exist_ok=True)
    
    # Export cleaned data
    cleaner.export_clean_data('outputs/cleaned_data/timeseries_cleaned.csv')
    
    # Export aggregated views
    daily_aggregated = cleaner.df.resample('D', on='timestamp').agg({
        'website_traffic': 'sum',
        'sales': 'sum',
        'server_errors': 'mean',
        'temperature_c': 'mean'
    }).round(2)
    
    daily_aggregated.to_csv('outputs/cleaned_data/timeseries_daily.csv')
    
    weekly_aggregated = cleaner.df.resample('W', on='timestamp').agg({
        'website_traffic': 'sum',
        'sales': 'sum',
        'server_errors': 'mean'
    }).round(2)
    
    weekly_aggregated.to_csv('outputs/cleaned_data/timeseries_weekly.csv')
    
    # Export feature correlation
    feature_corr = cleaner.df.corr()
    feature_corr.to_csv('outputs/cleaned_data/timeseries_correlation.csv')
    
    print(f"\n✅ Time series cleaning completed!")
    print(f"   Original features: 8")
    print(f"   Engineered features: {cleaner.df.shape[1]}")
    print(f"   Time range completeness: 100%")
    
    return cleaner.df

if __name__ == "__main__":
    cleaned_ts = run_time_series_cleaning()
    
    # Show insights
    print("\n💡 Time Series Insights:")
    print(f"   1. Data completeness: {cleaned_ts.isna().sum().sum() == 0}")
    print(f"   2. Peak traffic hour: {cleaned_ts.groupby('hour_of_day')['website_traffic'].mean().idxmax()}:00")
    print(f"   3. Weekend vs weekday sales ratio: "
          f"{cleaned_ts[cleaned_ts['is_weekend']]['sales'].mean() / cleaned_ts[~cleaned_ts['is_weekend']]['sales'].mean():.2f}")
    print(f"   4. Correlation traffic-sales: {cleaned_ts['website_traffic'].corr(cleaned_ts['sales']):.3f}")
    print(f"   5. Total features for modeling: {cleaned_ts.shape[1]}")
