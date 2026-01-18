"""
Test module transformers.py
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.transformers import DataTransformer, FeatureEngineer

class TestDataTransformer:
    """Test DataTransformer"""
    
    def setup_method(self):
        """Setup per ogni test"""
        self.transformer = DataTransformer()
        
        # Crea dataframe di test
        self.test_df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'age': [25, 30, None, 35],
            'income': [50000, 75000, 60000, None],
            'join_date': ['2023-01-01', '2023-02-15', '2023-03-20', '2023-04-10'],
            'category': ['A', 'B', 'A', 'C']
        })
    
    def test_normalize_column(self):
        """Test normalize column"""
        df = self.test_df.copy()
        result = self.transformer.normalize_column(df, 'age')
        
        # Verifica che i valori siano tra 0 e 1
        assert result['age_normalized'].min() >= 0
        assert result['age_normalized'].max() <= 1
    
    def test_encode_categorical(self):
        """Test encoding categorial"""
        df = self.test_df.copy()
        result = self.transformer.encode_categorical(df, 'category')
        
        # Verifica che siano state create colonne one-hot
        assert 'category_A' in result.columns
        assert 'category_B' in result.columns
        assert 'category_C' in result.columns
    
    def test_impute_missing_mean(self):
        """Test input missing values"""
        df = self.test_df.copy()
        result = self.transformer.impute_missing(df, 'age', strategy='mean')
        
        # Verifica che non ci siano NaN
        assert result['age'].isna().sum() == 0
        
        # Verifica che il valore imputato sia la media
        expected_mean = self.test_df['age'].mean()
        imputed_value = result.loc[2, 'age']  # Indice con NaN originale
        assert imputed_value == expected_mean
    
    def test_parse_dates(self):
        """Test parsing date"""
        df = self.test_df.copy()
        result = self.transformer.parse_dates(df, 'join_date')
        
        # Verifica che la colonna sia di tipo datetime
        assert pd.api.types.is_datetime64_any_dtype(result['join_date'])
        
        # Verifica formato
        assert result['join_date'].dt.year.iloc[0] == 2023
    
    def test_remove_outliers_iqr(self):
        """Test outliers con IQR"""
        # Crea dati con outliers
        data = pd.Series([1, 2, 3, 4, 5, 100])  # 100 è un outlier
        result = self.transformer.remove_outliers_iqr(data)
        
        # Verifica che l'outlier sia stato rimosso
        assert 100 not in result.values
        assert len(result) == 5

class TestFeatureEngineer:
    """Test FeatureEngineer"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engineer = FeatureEngineer()
        
        # Dati transazioni di test
        self.transactions = pd.DataFrame({
            'customer_id': ['C001', 'C001', 'C002', 'C003', 'C001'],
            'amount': [100.0, 200.0, 50.0, 300.0, 150.0],
            'date': pd.date_range('2023-01-01', periods=5),
            'product_category': ['Electronics', 'Books', 'Clothing', 'Electronics', 'Books']
        })
    
    def test_create_rfm_features(self):
        """Test creation feature RFM"""
        result = self.engineer.create_rfm_features(self.transactions)
        
        # Verifica che le colonne RFM esistano
        assert 'recency' in result.columns
        assert 'frequency' in result.columns
        assert 'monetary' in result.columns
        
        # Verifica valori
        assert len(result) == 3  # 3 clienti unici
    
    def test_create_time_based_features(self):
        """Test creation feature time based"""
        df = self.transactions.copy()
        result = self.engineer.create_time_based_features(df, 'date')
        
        # Verifica nuove colonne
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        
        # Verifica valori
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
    
    def test_create_aggregated_features(self):
        """Test feature aggregate per client"""
        result = self.engineer.create_customer_aggregates(self.transactions)
        
        # Verifica colonne aggregate
        assert 'total_spent' in result.columns
        assert 'avg_transaction' in result.columns
        assert 'transaction_count' in result.columns
        
        # Verifica calcoli
        customer_001 = result[result['customer_id'] == 'C001']
        assert customer_001['total_spent'].iloc[0] == 450.0  # 100+200+150
        assert customer_001['transaction_count'].iloc[0] == 3
    
    def test_create_interaction_features(self):
        """Test feature di interazione"""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })
        
        result = self.engineer.create_interaction_features(df, ['age', 'income'])
        
        # Verifica che sia stata creata la feature di interazione
        assert 'age_income_interaction' in result.columns
        
        # Verifica calcolo
        assert result['age_income_interaction'].iloc[0] == 25 * 50000
    
    def test_create_binned_features(self):
        """Test binning for feature continue"""
        df = pd.DataFrame({
            'age': [18, 25, 35, 45, 55, 65]
        })
        
        result = self.engineer.create_binned_features(df, 'age', bins=3)
        
        # Verifica nuova colonna categorica
        assert 'age_binned' in result.columns
        assert result['age_binned'].dtype == 'category'

class TestEcommerceTransformers:
    """Test for trasformers e-commerce"""
    
    def test_calculate_customer_lifetime_value(self):
        """Test calc CLV"""
        from src.transformers import EcommerceTransformer
        
        transformer = EcommerceTransformer()
        
        transactions = pd.DataFrame({
            'customer_id': ['C001', 'C001', 'C002'],
            'amount': [100, 200, 50],
            'profit_margin': [0.3, 0.4, 0.2]
        })
        
        clv_df = transformer.calculate_clv(transactions)
        
        assert 'clv' in clv_df.columns
        assert clv_df[clv_df['customer_id'] == 'C001']['clv'].iloc[0] == 110  # (100*0.3 + 200*0.4)
    
    def test_create_product_affinity_features(self):
        """Test feature di affinità prodotto"""
        from src.transformers import EcommerceTransformer
        
        transformer = EcommerceTransformer()
        
        transactions = pd.DataFrame({
            'customer_id': ['C001', 'C001', 'C002'],
            'product_id': ['P1', 'P2', 'P1'],
            'category': ['A', 'B', 'A']
        })
        
        result = transformer.create_product_affinity(transactions)
        
        assert 'preferred_category' in result.columns
        assert 'products_purchased' in result.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
