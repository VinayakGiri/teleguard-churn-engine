import os
import pandas as pd
import numpy as np

# Resolve the project root based on this file's location (src/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def clean_data(input_path=None, output_path=None):
    """
    Reads the raw Telco churn dataset, preprocesses it, and saves the cleaned version.
    Paths are resolved absolutely from the project root so this script works from any terminal location.
    """
    if input_path is None:
        input_path = os.path.join(PROJECT_ROOT, "data", "telco_raw.csv")
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "telco_cleaned.csv")

    print(f"Reading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 2. Convert TotalCharges column to numeric — coerce empty strings to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Drop all rows with any NaN values
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows containing NaN values.")
    
    # 4. Drop the customerID column since it is not useful for modeling
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        
    # 5. Convert the Churn column from Yes/No text to 1/0 numbers
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 6. Convert other Yes/No columns to 1/0 numbers
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    # 7. Use pandas get_dummies to encode all remaining categorical columns
    categorical_cols = [
        'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    
    # Only keep categorical columns that actually exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Encode with drop_first=True to avoid multicollinearity (dummy variable trap)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 8. Save the cleaned dataframe
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # 9. Print final statistics
    final_shape = df.shape
    num_features = final_shape[1] - 1  # Excluding the target 'Churn'
    churn_rate = (df['Churn'].sum() / len(df)) * 100
    
    print("\n" + "="*40)
    print("PREPROCESSING COMPLETE")
    print("="*40)
    print(f"Final Shape       : {final_shape[0]} rows, {final_shape[1]} columns")
    print(f"Number of Features: {num_features}")
    print(f"Churn Rate        : {churn_rate:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    clean_data()
