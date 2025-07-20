import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset
file_path = r"C:\Users\Maruf Ajimati\Documents\Nexford Assignments\BAN6800 Business Analytics Capstone\Milestone 1\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

# Step 2: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 3: Clean 'TotalCharges' and drop missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Step 4: Reset index
df.reset_index(drop=True, inplace=True)

# Step 5: Feature engineering
df['AvgChargesPerMonth'] = df['TotalCharges'] / df['tenure']
df['IsSenior'] = df['SeniorCitizen'].apply(lambda x: 1 if x == 1 else 0)

# Step 6: Encode categorical variables (excluding 'customerID')
categorical_cols = df.select_dtypes(include='object').columns.drop('customerID')
df_encoded = pd.get_dummies(df.drop(columns=['customerID']), columns=categorical_cols, drop_first=True)

# Step 7: Normalize numerical features
scaler = MinMaxScaler()
numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.drop(['SeniorCitizen', 'IsSenior'])
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# Step 8: Dimensionality reduction using PCA (keep 95% variance)
pca = PCA(n_components=0.95, random_state=42)
principal_components = pca.fit_transform(df_encoded[numeric_cols])
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
df_pca = pd.DataFrame(principal_components, columns=pca_columns)

# Step 9: Merge with non-numeric/other columns
other_cols = df_encoded.drop(columns=numeric_cols).reset_index(drop=True)
df_final = pd.concat([other_cols, df_pca], axis=1)

# Step 10: Visualize class imbalance (after encoding)
if 'Churn_Yes' in df_final.columns:
    churn_counts = df_final['Churn_Yes'].value_counts()
    churn_counts.plot(kind='bar', title='Churn Class Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
    plt.tight_layout()
    chart_path = r"C:\Users\Maruf Ajimati\Documents\Nexford Assignments\BAN6800 Business Analytics Capstone\Milestone 1\python_telco\churn_distribution.png"
    plt.savefig(chart_path)
    plt.close()
else:
    print("⚠️ Warning: 'Churn_Yes' column not found. Skipping chart.")

# Step 11: Save cleaned dataset
output_path = r"C:\Users\Maruf Ajimati\Documents\Nexford Assignments\BAN6800 Business Analytics Capstone\Milestone 1\python_telco\cleaned_telco_final.csv"
df_final.to_csv(output_path, index=False)

print("✅ Data preparation completed.")
print(f"Cleaned dataset saved to: {output_path}")
if 'chart_path' in locals():
    print(f"Churn distribution chart saved as: {chart_path}")
