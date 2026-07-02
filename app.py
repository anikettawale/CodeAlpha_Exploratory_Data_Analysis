import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("dataset.csv")

print("=" * 60)
print("       EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Display First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Check Duplicate Values
print("\nDuplicate Rows:", df.duplicated().sum())

# Correlation Matrix
numeric_df = df.select_dtypes(include=["number"])

if not numeric_df.empty:
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Histograms
numeric_df.hist(figsize=(10,8))
plt.suptitle("Histogram of Numerical Features")
plt.show()

# Box Plots
for column in numeric_df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot - {column}")
    plt.show()

print("\nEDA Completed Successfully!")
