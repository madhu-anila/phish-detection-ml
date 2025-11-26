import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_combined_emails.csv')

def check_class_imbalance(df, label_column='label'):
    """
    Comprehensive class imbalance check
    """
    print("="*60)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    # Basic counts
    class_counts = df[label_column].value_counts().sort_index()
    total = len(df)
    
    print(f"\nTotal Samples: {total}")
    print(f"\nClass Distribution:")
    for label, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  Class {label}: {count:,} samples ({percentage:.2f}%)")
    
    # Imbalance metrics
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = majority_class / minority_class
    
    print(f"\nImbalance Metrics:")
    print(f"  Majority Class Count: {majority_class:,}")
    print(f"  Minority Class Count: {minority_class:,}")
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Severity assessment
    if imbalance_ratio < 1.5:
        severity = "BALANCED"
    elif imbalance_ratio < 3:
        severity = "SLIGHTLY IMBALANCED"
    elif imbalance_ratio < 5:
        severity = "MODERATELY IMBALANCED"
    else:
        severity = "SEVERELY IMBALANCED"
    
    print(f"  Severity: {severity}")
    
    return class_counts

# Run the analysis
class_counts = check_class_imbalance(df)
