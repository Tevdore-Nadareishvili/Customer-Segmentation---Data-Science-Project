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
        