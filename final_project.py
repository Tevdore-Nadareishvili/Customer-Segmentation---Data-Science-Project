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
import matplotlib.pyplot as plt
import plotly.express as px



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
        
        print(f"[INFO] RFM Table Created. Shape: {self.rfm.shape}")
        return self.rfm


# ==========================================
# 2. VISUALIZATION CLASS
# ==========================================
class DataVisualizer:
    """
    Handles all static and interactive visualizations.
    Implements 6 distinct visualization types.
    """

    @staticmethod
    def plot_distributions(df, features):
        """Type 1: Histograms (Distribution Analysis)."""
        print("[INFO] Plotting Feature Distributions (Histogram)...")
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(features, 1):
            plt.subplot(1, 3, i)
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_boxplots(df, features):
        """Type 2: Box Plots (Outlier Detection)."""
        print("[INFO] Plotting Box Plots for Outlier Detection...")
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(features, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(y=df[col], color='lightgreen')
            plt.title(f'Box Plot of {col}')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_correlation(df):
        """Type 3: Heatmap (Correlation Analysis)."""
        print("[INFO] Plotting Correlation Matrix (Heatmap)...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()


    @staticmethod
    def plot_pairplot(df):
        """Type 4: Pair Plot (Multivariate Analysis)."""
        print("[INFO] Plotting Pairplot...")
        sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
        plt.show()

    
    @staticmethod
    def plot_cluster_counts(df, cluster_col):
        """Type 5: Bar Chart (Cluster Size Analysis)."""
        print("[INFO] Plotting Cluster Sizes (Bar Chart)...")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=cluster_col, data=df, palette='viridis')
        plt.title('Number of Customers per Cluster')
        plt.show()


    
    @staticmethod
    def plot_3d_clusters(df, x, y, z, cluster_col):
        """Type 6: 3D Scatter (Interactive - Bonus)."""
        print("[INFO] Generating Interactive 3D Cluster Plot...")
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, color=cluster_col,
            title="3D Customer Segments (RFM)",
            labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency', 'Monetary': 'Total Spend'},
            opacity=0.7
        )
        fig.show()

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
def main():
    print("=== FINAL PROJECT: Customer Segmentation Pipeline ===")
    
    try:
        # 1. Data Processing
        DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        LOCAL_FILE = "online_retail.xlsx"
        
        processor = DataProcessor(url=DATA_URL, filepath=LOCAL_FILE)
        df = processor.load_data()
        df_clean = processor.clean_data()
        rfm_df = processor.generate_rfm_features()
		
        print(df)
        print(df_clean)
        print("\n=== Project Execution Complete ===")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()


    # 2. EDA (Implementing 5+ Visualization Types)
        print("\n=== Phase 2: Exploratory Data Analysis ===")
        viz = DataVisualizer()
        viz.plot_distributions(rfm_df, ['Recency', 'Frequency', 'Monetary'])
        viz.plot_boxplots(rfm_df, ['Recency', 'Frequency', 'Monetary'])
        viz.plot_correlation(rfm_df)

if __name__ == "__main__":
    main()