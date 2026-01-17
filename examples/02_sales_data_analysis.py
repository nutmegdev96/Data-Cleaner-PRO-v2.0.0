"""
Example 2: Sales Data Analysis Pipeline
Cleaning and preparing sales data for analysis and reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append('../src')
from cleaner import DataCleaner
from transformers import DataTransformer
from utils import calculate_data_quality_score, create_data_dictionary

print("=" * 60)
print("📈 EXAMPLE 2: Sales Data Analysis Pipeline")
print("=" * 60)

def generate_sample_sales_data():
    """Generate realistic sales data with seasonality and trends"""
    np.random.seed(42)
    n_rows = 5000
    
    # Generate dates
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports', 'Toys']
    
    # Regions
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa']
    
    data = {
        'OrderID': [f'ORD{10000 + i}' for i in range(n_rows)],
        'OrderDate': dates,
        'Product Category': np.random.choice(categories + [None] * 2, n_rows),
        'Product Name': [f'Product_{i%100}' for i in range(n_rows)],
        'Quantity': np.random.poisson(3, n_rows),
        'Unit Price': np.random.uniform(10, 500, n_rows).round(2),
        'Discount %': np.random.choice([0, 5, 10, 15, 20, 25, None], n_rows, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]),
        'Customer Region': np.random.choice(regions + [None], n_rows),
        'Sales Rep ID': [f'REP{np.random.randint(1, 21):03d}' if np.random.random() > 0.1 else None for _ in range(n_rows)],
        'Payment Method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'Cash', None], n_rows),
        'Shipping Cost': np.random.exponential(10, n_rows).round(2),
        'Order Status': np.random.choice(['Delivered', 'Pending', 'Shipped', 'Cancelled', 'Returned'], n_rows),
        'Customer Rating': np.random.choice([1, 2, 3, 4, 5, None], n_rows, p=[0.05, 0.1, 0.15, 0.3, 0.35, 0.05]),
    }
    
    # Calculate total amount (with some errors)
    data['Total Amount'] = []
    for i in range(n_rows):
        quantity = data['Quantity'][i]
        price = data['Unit Price'][i]
        discount = data['Discount %'][i] or 0
        shipping = data['Shipping Cost'][i]
        
        # Introduce some calculation errors
        if np.random.random() < 0.02:
            # Wrong calculation
            total = quantity * price + shipping
        else:
            # Correct calculation
            total = quantity * price * (1 - discount/100) + shipping
        
        data['Total Amount'].append(round(total, 2))
    
    df = pd.DataFrame(data)
    
    # Introduce outliers
    outlier_idx = np.random.choice(n_rows, 30, replace=False)
    df.loc[outlier_idx, 'Total Amount'] = df.loc[outlier_idx, 'Total Amount'] * np.random.uniform(5, 20, 30)
    
    # Introduce negative quantities (data entry errors)
    negative_idx = np.random.choice(n_rows, 15, replace=False)
    df.loc[negative_idx, 'Quantity'] = -df.loc[negative_idx, 'Quantity']
    
    # Save data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_sales.csv', index=False)
    df.to_excel('data/sample_sales.xlsx', index=False)
    
    return df

def run_sales_analysis_example():
    """Complete sales data cleaning and analysis pipeline"""
    
    print("\n📥 Step 1: Loading sales data...")
    if os.path.exists('data/sample_sales.csv'):
        df = pd.read_csv('data/sample_sales.csv', parse_dates=['OrderDate'])
    else:
        df = generate_sample_sales_data()
    
    print(f"   Sales data loaded: {df.shape[0]} orders, {df.shape[1]} columns")
    
    # Initialize cleaner
    cleaner = DataCleaner(df, verbose=True)
    
    print("\n🔧 Step 2: Data cleaning pipeline...")
    
    # Custom cleaning for sales data
    cleaner.standardize_column_names(case='snake') \
          .handle_missing_values(strategy='auto', threshold=0.15) \
          .convert_data_types({
              'order_date': 'datetime64[ns]',
              'quantity': 'int',
              'unit_price': 'float',
              'discount_%': 'float',
              'total_amount': 'float'
          }) \
          .remove_duplicates(subset=['order_id']) \
          .detect_outliers(method='iqr', threshold=2.0)
    
    # Fix data quality issues specific to sales
    print("\n🎯 Step 3: Fixing business logic issues...")
    
    # 1. Ensure positive quantities
    negative_qty_mask = cleaner.df['quantity'] < 0
    if negative_qty_mask.any():
        print(f"   Found {negative_qty_mask.sum()} orders with negative quantity")
        cleaner.df.loc[negative_qty_mask, 'quantity'] = abs(cleaner.df.loc[negative_qty_mask, 'quantity'])
    
    # 2. Validate discount range
    invalid_discount_mask = (cleaner.df['discount_%'] < 0) | (cleaner.df['discount_%'] > 100)
    if invalid_discount_mask.any():
        print(f"   Found {invalid_discount_mask.sum()} invalid discounts")
        cleaner.df.loc[invalid_discount_mask, 'discount_%'] = cleaner.df.loc[invalid_discount_mask, 'discount_%'].clip(0, 100)
    
    # 3. Recalculate total amount if calculation seems wrong
    expected_total = (cleaner.df['quantity'] * cleaner.df['unit_price'] * 
                     (1 - cleaner.df['discount_%'].fillna(0) / 100) + 
                     cleaner.df['shipping_cost'])
    
    calculation_error = abs(cleaner.df['total_amount'] - expected_total) > 0.01
    if calculation_error.any():
        print(f"   Found {calculation_error.sum()} calculation errors")
        cleaner.df.loc[calculation_error, 'total_amount'] = expected_total[calculation_error]
    
    print("\n🔄 Step 4: Feature engineering for analysis...")
    transformer = DataTransformer(verbose=True)
    
    # Extract datetime features
    cleaner.df = transformer.extract_datetime_features(cleaner.df, 'order_date')
    
    # Create new features
    cleaner.df = transformer.create_features(cleaner.df, {
        'discounted_price': ['unit_price * (1 - discount_% / 100)'],
        'profit_margin': ['(total_amount - (quantity * unit_price * 0.6)) / total_amount'],  # Assume 40% margin
        'order_size_category': [
            "np.where(total_amount < 100, 'Small', " +
            "np.where(total_amount < 500, 'Medium', 'Large'))"
        ],
        'is_weekend_order': ['order_date_weekday >= 5'],
        'monthly_segment': [
            "np.where(order_date_month.isin([11, 12]), 'Holiday', " +
            "'Regular')"
        ]
    })
    
    # Reduce skewness in monetary columns
    cleaner.df = transformer.handle_skewness(cleaner.df, 
                                            ['total_amount', 'unit_price'], 
                                            threshold=0.5)
    
    print("\n📊 Step 5: Generate sales analytics...")
    
    # Monthly sales summary
    monthly_sales = cleaner.df.groupby('order_date_month').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_rating': 'mean'
    }).round(2)
    
    monthly_sales.columns = ['total_sales', 'avg_order_value', 'order_count', 
                            'total_quantity', 'avg_rating']
    
    print("\n📅 Monthly Sales Summary:")
    print(monthly_sales)
    
    # Category performance
    category_performance = cleaner.df.groupby('product_category').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_rating': 'mean'
    }).round(2)
    
    category_performance.columns = ['category_revenue', 'avg_order_value', 
                                   'order_count', 'total_quantity', 'avg_rating']
    
    print("\n🏷️  Category Performance:")
    print(category_performance.sort_values('category_revenue', ascending=False))
    
    # Regional analysis
    regional_sales = cleaner.df.groupby('customer_region').agg({
        'total_amount': 'sum',
        'order_id': 'count',
        'customer_rating': 'mean'
    }).round(2)
    
    regional_sales.columns = ['regional_revenue', 'order_count', 'avg_rating']
    regional_sales['revenue_share'] = (regional_sales['regional_revenue'] / 
                                      regional_sales['regional_revenue'].sum() * 100).round(1)
    
    print("\n🌍 Regional Sales Analysis:")
    print(regional_sales.sort_values('regional_revenue', ascending=False))
    
    print("\n📈 Step 6: Export results for reporting...")
    os.makedirs('outputs/cleaned_data', exist_ok=True)
    
    # Export cleaned data
    cleaner.export_clean_data('outputs/cleaned_data/sales_cleaned.xlsx')
    
    # Export analytics
    monthly_sales.to_csv('outputs/cleaned_data/monthly_sales.csv')
    category_performance.to_csv('outputs/cleaned_data/category_performance.csv')
    regional_sales.to_csv('outputs/cleaned_data/regional_sales.csv')
    
    # Data dictionary
    data_dict = create_data_dictionary(cleaner.df, {
        'order_id': 'Unique order identifier',
        'order_date': 'Date when order was placed',
        'total_amount': 'Final amount paid by customer',
        'product_category': 'Category of purchased product',
        'profit_margin': 'Estimated profit margin on order',
        'order_size_category': 'Size classification based on order value'
    })
    
    data_dict.to_csv('outputs/cleaned_data/sales_data_dictionary.csv', index=False)
    
    # Data quality score
    quality_score = calculate_data_quality_score(cleaner.df)
    print(f"\n✅ Sales Data Quality Score: {quality_score}/100")
    
    print("\n🎉 Sales data analysis pipeline completed!")
    print("   Files saved in 'outputs/cleaned_data/'")
    
    return cleaner.df

if __name__ == "__main__":
    cleaned_sales = run_sales_analysis_example()
    
    # Show insights
    print("\n💡 Key Insights:")
    print(f"   1. Total Orders: {len(cleaned_sales):,}")
    print(f"   2. Total Revenue: ${cleaned_sales['total_amount'].sum():,.2f}")
    print(f"   3. Average Order Value: ${cleaned_sales['total_amount'].mean():.2f}")
    print(f"   4. Top Category: {cleaned_sales['product_category'].mode().iloc[0]}")
    print(f"   5. Data Quality Score: {calculate_data_quality_score(cleaned_sales)}/100")
