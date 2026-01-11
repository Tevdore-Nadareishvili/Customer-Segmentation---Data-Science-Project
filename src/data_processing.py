import os
import io
import requests
import pandas as pd
import numpy as np

class DataProcessor:
    """
    Handles data loading, cleaning, and feature engineering.
    """
    
    def __init__(self, url=None, raw_filepath=None, processed_filepath=None):
        self.url = url
        self.raw_filepath = raw_filepath
        self.processed_filepath = processed_filepath
        self.df = None
        self.rfm = None

    def load_data(self):
        """Loads data from local file or downloads it if not present."""
        try:
            if self.raw_filepath and os.path.exists(self.raw_filepath):
                print(f"[INFO] Loading data from {self.raw_filepath}...")
                if self.raw_filepath.endswith('.csv'):
                    self.df = pd.read_csv(self.raw_filepath, encoding='ISO-8859-1')
                else:
                    self.df = pd.read_excel(self.raw_filepath)
            elif self.url:
                print(f"[INFO] Downloading data from {self.url}...")
                response = requests.get(self.url)
                response.raise_for_status()
                with io.BytesIO(response.content) as f:
                    self.df = pd.read_excel(f)
                
                # Save raw file
                if self.raw_filepath:
                    os.makedirs(os.path.dirname(self.raw_filepath), exist_ok=True)
                    self.df.to_excel(self.raw_filepath, index=False)
            else:
                raise ValueError("No data source provided.")
            
            print(f"[INFO] Initial Data Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            raise Exception(f"Failed to load data: {e}")

    def clean_data(self):
        """
        Performs data cleaning: removing missing IDs, cancellations, and invalid values.
        """
        
        if self.df is None: 
             raise ValueError("Data not loaded.")

        print("[INFO] Starting Data Cleaning...")
        
        # 1. Drop missing CustomerID
        self.df.dropna(subset=['CustomerID'], inplace=True)
        
        # 2. Remove cancelled transactions (InvoiceNo starts with 'C')
        self.df['InvoiceNo'] = self.df['InvoiceNo'].astype(str)
        self.df = self.df[~self.df['InvoiceNo'].str.contains('C')]

		# 3. Remove negative quantities/prices
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        
		# 4. Convert Date
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
		# 5. Calculate Total Price
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        
        print(f"[INFO] Shape after cleaning: {self.df.shape}")
        return self.df

    def generate_rfm_features(self):
        """
        Bonus Feature Engineering: Creates Recency, Frequency, and Monetary (RFM) table.
        """
        print("[INFO] Generating RFM Features...")
        
		# Reference date: 1 day after the last transaction
        snapshot_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)
        
		# Aggregation
        self.rfm = self.df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
            'InvoiceNo': 'nunique',                                   # Frequency
            'TotalPrice': 'sum'                                       # Monetary
        })
        
        self.rfm.rename(columns={
            'InvoiceDate': 'Recency',
            'InvoiceNo': 'Frequency',
            'TotalPrice': 'Monetary'
        }, inplace=True)
        
		# Statistical Outlier Handling
        # We remove extreme outliers to ensure the K-Means model is stable.
        # Thresholds chosen based on domain knowledge and initial statistical review.
        print("[INFO] Handling Outliers...")
        initial_shape = self.rfm.shape
        self.rfm = self.rfm[self.rfm['Monetary'] < 25000] 
        self.rfm = self.rfm[self.rfm['Frequency'] < 300]
        print(f"[INFO] Outliers removed. Rows dropped: {initial_shape[0] - self.rfm.shape[0]}")
        
        # Save processed data
        if self.processed_filepath:
            os.makedirs(os.path.dirname(self.processed_filepath), exist_ok=True)
            self.rfm.to_csv(self.processed_filepath)
            
        print(f"[INFO] RFM Table Created. Shape: {self.rfm.shape}")
        return self.rfm