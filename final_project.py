"""
Data Science with Python - Final Project
Project: E-Commerce Customer Segmentation & Analysis
Team Members: Valer Murtskhvaladze, Tevdore Nadareishvili, Nika Tamliani, Erekle Khomasuridze
"""

import pandas as pd
import seaborn as sns
import requests
import io
import sys
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42


# ==========================================
# 1. DATA LOADING & PROCESSING CLASS
# ==========================================
class DataProcessor:
    """
    Handles data loading, cleaning, and feature engineering.
    """
    
    def __init__(self, url=None, filepath=None):
        self.url = url
        self.filepath = filepath
        self.df = None
        self.rfm = None
        
    def load_data(self):
        """Loads data from local file or downloads it if not present."""
        try:
            if self.filepath and os.path.exists(self.filepath):
                print(f"[INFO] Loading data from {self.filepath}...")
                if self.filepath.endswith('.csv'):
                    self.df = pd.read_csv(self.filepath, encoding='ISO-8859-1')
                else:
                    self.df = pd.read_excel(self.filepath)
            elif self.url:
                print(f"[INFO] Downloading data from {self.url} (This may take a moment)...")
                response = requests.get(self.url)
                response.raise_for_status()
                with io.BytesIO(response.content) as f:
                    self.df = pd.read_excel(f)

                if self.filepath:
                    self.df.to_excel(self.filepath, index=False)
            else:
                raise ValueError("No data source provided.")
            
            print(f"[INFO] Initial Data Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            sys.exit(1)
	
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
    
	

def main():
    print("=== FINAL PROJECT: Customer Segmentation Pipeline ===")
    
    try:
        # 1. Data Processing
        DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        LOCAL_FILE = "online_retail.xlsx"
        
        processor = DataProcessor(url=DATA_URL, filepath=LOCAL_FILE)
        df = processor.load_data()
        df_clean = processor.clean_data()
        
        print(df)
        print(df_clean)
        print("\n=== Project Execution Complete ===")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()