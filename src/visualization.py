import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

class DataVisualizer:
    """
    Handles static and interactive visualizations and saves them.
    """
    
    def __init__(self, output_dir='reports/figures'):
        # Initialize the visualizer with an output directory
        # Create the directory if it does not exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plot(self, filename):
        # Helper function to save the current matplotlib figure to disk
        # Closes the figure afterwards to free up memory
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        print(f"[INFO] Plot saved to {path}")
        plt.close()

    def plot_distributions(self, df, features):
        """Type 1: Histograms (Distribution Analysis)."""
        print("[INFO] Plotting Feature Distributions (Histogram)...")
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(features, 1):
            plt.subplot(1, 3, i)
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        self.save_plot('01_distributions.png')

    def plot_boxplots(self, df, features):
        """Type 2: Box Plots (Outlier Detection)."""
        print("[INFO] Plotting Box Plots for Outlier Detection...")
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(features, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(y=df[col], color='lightgreen')
            plt.title(f'Box Plot of {col}')
        plt.tight_layout()
        self.save_plot('02_boxplots.png')

    def plot_correlation(self, df):
        """Type 3: Heatmap (Correlation Analysis)."""
        print("[INFO] Plotting Correlation Matrix (Heatmap)...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        self.save_plot('03_correlation_matrix.png')

    def plot_cluster_counts(self, df, cluster_col):
        """Type 4: Bar Chart (Cluster Size Analysis)."""
        print("[INFO] Plotting Cluster Sizes (Bar Chart)...")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=cluster_col, data=df, hue=cluster_col, palette='viridis', legend=False)
        plt.title('Number of Customers per Cluster')
        self.save_plot('04_cluster_counts.png')

    def plot_3d_clusters(self, df, x, y, z, cluster_col):
        """Type 5: 3D Scatter (Interactive)."""
        print("[INFO] Generating Interactive 3D Cluster Plot...")
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, color=cluster_col,
            title="3D Customer Segments (RFM)",
            labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency', 'Monetary': 'Total Spend'},
            opacity=0.7
        )
        output_path = os.path.join(self.output_dir, '05_3d_clusters.html')
        fig.write_html(output_path)
        print(f"[INFO] Interactive 3D plot saved to {output_path}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Visualizes the confusion matrix."""
        # Visualizes model performance by showing True Positives vs False Positives/Negatives
        print(f"[INFO] Plotting Confusion Matrix for {model_name}...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        self.save_plot(f'06_confusion_matrix_{model_name.replace(" ", "_")}.png')

    def plot_feature_importance(self, model, feature_names):
        """Visualizes feature importance for Random Forest."""
        # Visualizes which features (R, F, or M) were most important for the Random Forest predictions
        print("[INFO] Plotting Feature Importance...")
        importances = model.feature_importances_
        indices = range(len(importances))
        
        plt.figure(figsize=(8, 5))
        plt.bar(indices, importances, align='center', color='teal')
        plt.xticks(indices, feature_names)
        plt.title('Feature Importance (Random Forest)')
        plt.ylabel('Importance Score')
        self.save_plot('07_feature_importance.png')
        
    def plot_monthly_sales(self, df):
        """
        Visualizes Spending Patterns over time (Monthly Sales).
        """
        print("[INFO] Plotting Monthly Sales Trends...")
        
        # Ensure Date is datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Group by Month (Y-M)
        # We use to_period('M') to handle Year-Month sorting correctly
        monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
        
        # Convert index back to string for plotting
        monthly_sales.index = monthly_sales.index.astype(str)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o', color='purple')
        plt.title('Total Revenue per Month (Spending Patterns)')
        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        self.save_plot('09_monthly_spending_patterns.png')

    def plot_country_distribution(self, df):
        """
        Visualizes Customer Demographics.
        """
        print("[INFO] Plotting Customer Demographics (Country)...")
        
        # Get Top 10 Countries by number of unique customers
        country_counts = df.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=country_counts.values, y=country_counts.index, palette='magma')
        plt.title('Top 10 Countries by Customer Count (Demographics)')
        plt.xlabel('Number of Unique Customers')
        plt.ylabel('Country')
        plt.tight_layout()
        self.save_plot('10_country_demographics.png')