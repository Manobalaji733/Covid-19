# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"
df = pd.read_csv(url)

# Display basic info
print("Dataset Preview:")
print(df.head())

# Handle missing values (Fill NaN with 0 for simplicity)
df.fillna(0, inplace=True)

# Step 1: Data Aggregation - Group by 'continent' and compute max values
grouped_data = df.groupby('continent').agg({
    'total_cases': 'max',
    'total_deaths': 'max',
    'gdp_per_capita': 'max'
}).reset_index()

# Step 2: Create a New Feature - Deaths-to-Cases Ratio
grouped_data['total_deaths_to_total_cases'] = (
    grouped_data['total_deaths'] / grouped_data['total_cases']
).replace([np.inf, np.nan], 0)  # Handle division by zero cases

# Display processed data
print("\nProcessed Data:")
print(grouped_data)

# Step 3: Data Visualization

# 3.1 Histogram - GDP per Capita Distribution
plt.figure(figsize=(8, 5))
sns.histplot(grouped_data['gdp_per_capita'], kde=True, color='blue')
plt.title('GDP per Capita Distribution')
plt.xlabel('GDP per Capita')
plt.ylabel('Frequency')
plt.show()

# 3.2 Scatter Plot - Total Cases vs Death-to-Case Ratio
plt.figure(figsize=(8, 5))
sns.scatterplot(x='total_cases', y='total_deaths_to_total_cases', data=grouped_data, color='red')
plt.xlabel('Total Cases')
plt.ylabel('Death-to-Case Ratio')
plt.title('Death-to-Case Ratio vs Total Cases')
plt.show()

# 3.3 Pairplot - Checking Relationships Between Variables
sns.pairplot(grouped_data, diag_kind='kde')
plt.show()

# 3.4 Bar Plot - Total Cases by Continent
plt.figure(figsize=(8, 5))
sns.barplot(x='continent', y='total_cases', data=grouped_data, palette='viridis')
plt.title('Total Cases by Continent')
plt.xlabel('Continent')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.show()

# Save final results
grouped_data.to_csv("processed_covid_data.csv", index=False)
print("\nProcessed data saved to 'processed_covid_data.csv'")
