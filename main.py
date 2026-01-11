import os
import sys

# Import from the src package
from src.data_processing import DataProcessor
from src.visualization import DataVisualizer
from src.models import ModelTrainer

class ReportLogger:
    """Helper class to log outputs to both console and a file simultaneously."""
    def __init__(self, filepath):
        self.filepath = filepath
        # Clear file on init
        with open(self.filepath, 'w') as f:
            f.write("=== FINAL PROJECT EXECUTION LOG ===\n\n")

    def log(self, message):
        """Prints to console AND appends to file."""
        print(message)  # Print to console
        with open(self.filepath, 'a') as f:
            f.write(str(message) + "\n")

def main():
    # Setup directories
    os.makedirs('reports/results', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    # Initialize Logger
    logger = ReportLogger('reports/results/full_execution_log.txt')
    logger.log("=== FINAL PROJECT: Customer Segmentation Pipeline ===")

    try:
        # 1. Data Processing
        logger.log("\n--- Phase 1: Data Processing ---")
        DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        
        processor = DataProcessor(
            url=DATA_URL, 
            raw_filepath='data/raw/online_retail.xlsx',
            processed_filepath='data/processed/rfm_data.csv'
        )
        
        # Log data shape
        df = processor.load_data()
        logger.log(f"[INFO] Initial Data Shape: {df.shape}")
        
        df_clean = processor.clean_data()
        logger.log(f"[INFO] Shape after cleaning: {df_clean.shape}")
        
        rfm_df = processor.generate_rfm_features()
        logger.log(f"[INFO] Processed Data Shape: {rfm_df.shape}")
        

        # 2. EDA
        logger.log("\n--- Phase 2: Visualization ---")
        viz = DataVisualizer(output_dir='reports/figures')
        viz.plot_distributions(rfm_df, ['Recency', 'Frequency', 'Monetary'])
        viz.plot_boxplots(rfm_df, ['Recency', 'Frequency', 'Monetary'])
        viz.plot_correlation(rfm_df)
        viz.plot_monthly_sales(df_clean)       
        viz.plot_country_distribution(df_clean) 

        # 3. Machine Learning
        logger.log("\n--- Phase 3: Machine Learning ---")
        trainer = ModelTrainer(rfm_df)
        trainer.preprocess()
        
        # Save Elbow Plot
        trainer.determine_optimal_k(max_k=10, save_path='reports/figures/00_elbow_plot.png')

        # Clustering
        clustered_df = trainer.perform_clustering(n_clusters=4)
        
        # Log Cluster Summary
        logger.log("\n--- Business Insight: Cluster Summary ---")
        summary = clustered_df.groupby(['Cluster', 'Segment Name']).agg({
            'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'Cluster': 'count'
        }).rename(columns={'Cluster': 'Count'})
        logger.log(summary)

        # Cluster Visualizations
        viz.plot_cluster_counts(clustered_df, 'Segment Name')
        viz.plot_3d_clusters(clustered_df, 'Recency', 'Frequency', 'Monetary', 'Segment Name')

        # 4. Classification
        logger.log("\n--- Phase 4: Classification Modeling ---")
        logger.log("Objective: Predict Customer Segment based on RFM metrics.")
        
        results = trainer.train_classifiers()
        
        # Log Full Model Reports to File
        logger.log("\n=== Final Model Performance Reports ===")
        for name, metrics in results.items():
            logger.log(f"\nModel: {name}")
            logger.log(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.log("Classification Report:")
            logger.log(metrics['report'])
            
        # === Generate Classification Visualizations ===
        logger.log("\n=== Generating Classification Visualizations ===")
        for name, data in results.items():
            # 1. Plot Confusion Matrix for every model
            if 'y_test' in data and 'y_pred' in data:
                 viz.plot_confusion_matrix(data['y_test'], data['y_pred'], name)
            
            # 2. Plot Feature Importance ONLY for Random Forest
            if name == "Random Forest" and 'model' in data:
                viz.plot_feature_importance(data['model'], ['Recency', 'Frequency', 'Monetary'])
        # ===================================================
        
        # Final Summary
        logger.log("\n=== Final Project Report ===")
        logger.log(f"1. Data Size: {len(rfm_df)} customers processed.")
        logger.log(f"2. Clusters Identified: 4 Distinct Segments.")
        logger.log(f"3. Segments Found: {list(trainer.cluster_names.values())}")
        logger.log("4. Best Predictive Model: Random Forest")
        for model_name, metrics in results.items():
            logger.log(f"   - {model_name}: {metrics['accuracy']*100:.2f}% Accuracy")
            
        logger.log("\n=== Project Execution Complete ===")

    except Exception as e:
        logger.log(f"\n[CRITICAL ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()