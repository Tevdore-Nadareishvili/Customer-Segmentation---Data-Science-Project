import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

class ModelTrainer:
    """
    Handles K-Means Clustering and Classification Models.
    """
    
    def __init__(self, data, random_state=42):
        self.data = data
        self.X_scaled = None
        self.kmeans = None
        self.labels = None
        self.cluster_names = {}
        self.random_state = random_state
        
    def preprocess(self):
        """Log transform (to handle skew) and Scale data."""
        data_log = np.log1p(self.data)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(data_log)
        return self.X_scaled

    def determine_optimal_k(self, max_k=10, save_path=None):
        """Calculates inertia for Elbow Method and saves plot."""
        print("[INFO] Determining optimal K using Elbow Method...")
        inertia = []
        K_range = range(1, max_k + 1)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(self.X_scaled)
            inertia.append(km.inertia_)
            
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertia, marker='o', linestyle='--')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"[INFO] Elbow plot saved to {save_path}")
            plt.close()

    def perform_clustering(self, n_clusters=4):
        """
        Required: Unsupervised Learning (K-Means).
        """
        print(f"[INFO] Performing K-Means Clustering with k={n_clusters}...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X_scaled)
        self.data['Cluster'] = self.labels
        
        score = silhouette_score(self.X_scaled, self.labels)
        print(f"[INFO] Silhouette Score: {score:.4f}")
        
        # Auto-Label Clusters
        cluster_means = self.data.groupby('Cluster').mean()
        
        # Calculate global means for comparison
        global_r = self.data['Recency'].mean()
        global_f = self.data['Frequency'].mean()
        global_m = self.data['Monetary'].mean()
        
        for cluster_id, row in cluster_means.iterrows():
            is_recent = row['Recency'] < global_r
            is_frequent = row['Frequency'] > global_f
            is_big_spender = row['Monetary'] > global_m
            
            if is_recent and is_frequent and is_big_spender:
                label = "Champions (VIP)"
            elif is_recent and is_big_spender and not is_frequent:
                label = "Loyal Customers"
            elif is_recent and not is_big_spender:
                label = "Recent Low Spenders"
            elif not is_recent and (is_frequent or is_big_spender):
                label = "At Risk"
            else:
                label = "Lost/Dormant"
                
            self.cluster_names[cluster_id] = label
            
        self.data['Segment Name'] = self.data['Cluster'].map(self.cluster_names)
        print("[INFO] Cluster Auto-Labeling Complete:")
        print(self.cluster_names)
        return self.data

    def train_classifiers(self):
        """
        Required + Bonus: Train multiple classifiers to predict segments.
        """
        print("[INFO] Training Classification Models...")
        X = self.data[['Recency', 'Frequency', 'Monetary']]
        y = self.data['Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=5000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        results = {}
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            print(f"Accuracy: {acc:.4f}")
            print("Classification Report:\n", report)
            
            results[name] = {
                'model': model,
                'accuracy': acc, 
                'report': report,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
        return results