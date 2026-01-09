"""
Data Science with Python - Final Project
Project: E-Commerce Customer Segmentation & Analysis
Team Members: Valer Murtskhvaladze, Tevdore Nadareishvili, Nika Tamliani, Erekle Khomasuridze
"""

import pandas as pd
import seaborn as sns

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