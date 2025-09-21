# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset

df = pd.read_csv('data.csv')

# outlier removal function

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

# print initial data insights

print("head >\n", df.head())
print("\ninfo >\n", df.info())
print("\ndescribe >\n", df.describe())
print("\nshape >", df.shape)

# ---- Duplicate Removal ----

dups = df.duplicated()
print("\nNumber of duplicate rows:", dups.sum())
if dups.sum() > 0:
    df.drop_duplicates(inplace=True)
    print("\nDuplicates removed. New shape:", df.shape)

# ---- Outlier treatment ----

print("\nChecking for outliers using boxplots...")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.title("Before Outlier Removal")
plt.show()

for col in df.select_dtypes(include=[np.number]).columns:
    df = remove_outliers(df, col)
    print(f"\nOutliers removed from {col}. New shape:", df.shape)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(12,6))
sns.boxplot(data=df[numeric_cols])
plt.xticks(rotation=45)
plt.title("After Outlier Removal")
plt.show()

# ---- Missing Value Treatment ----

print("\nMissing values in each column(Before Treatment):\n", df.isnull().sum())

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"\nFilled missing values in numeric column '{col}' with median: {median_value}")
        else:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"\nFilled missing values in categorical column '{col}' with mode: {mode_value}")

print("\nMissing values in each column(After Treatment):\n", df.isnull().sum())

# ---- Univariate Analysis ----


numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(15, 10))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols)//3 + 1, 3, i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"{col} Distribution")

plt.tight_layout()
plt.show()

categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("Numeric Summary:\n", df[numeric_cols].describe())
print("\nCategorical Summary:\n", df[categorical_cols].describe())

# ---- Bivariate Analysis ----

sns.pairplot(df[numeric_cols])
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
