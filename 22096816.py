
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Path to your CSV file
file_path = r"API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv"

# Open and read the file to inspect its content
with open(file_path, 'r') as file:
    for i in range(5):  # Display the first 5 lines
        line = file.readline()
        print(line)

# Path to your CSV file
file_path = r"API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv"

# Open and read the file to inspect its content, skipping the initial characters
with open(file_path, 'r', encoding='utf-8-sig') as file:
    # Skip the initial lines with encoding characters
    for i in range(3):
        next(file)

    # Display the next 5 lines after skipping the initial lines
    for i in range(5):
        line = file.readline()
        print(line)


# File path
file_path = r"API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv"

# Load the dataset skipping the first few rows of metadata
data = pd.read_csv(file_path, skiprows=4)

# Display the first few rows of the dataset
print(data.head())

# Selecting relevant columns
relevant_columns = ['Country Name'] + [str(year) for year in range(1990, 2023)]
relevant_columns += ['Indicator Name']  # Add 'Indicator Name' column

# Creating a new DataFrame with relevant columns
relevant_data = data[relevant_columns]

# Displaying the first few rows of the new DataFrame
print(relevant_data.head())

# Check for missing values in the dataset
missing_values = relevant_data.isnull().sum()
print(missing_values)


# Select numeric columns and drop non-numeric ones
numeric_data = relevant_data.select_dtypes(include='number')

# Impute missing values with mean for numeric columns only
imputer = SimpleImputer(strategy='mean')
data_for_clustering_imputed = imputer.fit_transform(numeric_data)

# Normalizing the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_for_clustering_imputed)

# Apply k-means clustering with different numbers of clusters
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(normalized_data)
    inertia.append(kmeans.inertia_)


# Plotting the elbow method to find the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming you already have your data and clustering results in normalized_data and kmeans

# Fit KMeans with the chosen number of clusters
n_clusters = 4  # Change this to your chosen number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_data)

# Plotting the clustered data points
plt.figure(figsize=(8, 6))

# Scatter plot for the first two dimensions of the normalized data
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5, label='Data Points')

# Plotting centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering')
plt.legend()
plt.show()

# Display column names
print(data.columns)

# Define the columns of interest (years range from 1960 to 2022)
years_columns = [str(year) for year in range(1960, 2023)]

# Filter the DataFrame to keep only necessary columns
country_data_years = data[years_columns]

# Drop rows with any NaN values in the selected year columns
country_data_cleaned = country_data_years.dropna()

# Prepare the data for curve fitting
years = range(1960, 2023)  # Assuming years range from 1960 to 2022
values = country_data_cleaned.values.flatten()  # Flatten the selected values

# Plotting and curve fitting can follow using the 'years' and 'values' variables

import pandas as pd

# Replace the file path with your actual file location
file_path = r"API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv"

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path, skiprows=4)  # Skip the first 4 rows as they contain metadata

# Display the first few rows of the DataFrame
print(data.head())

# Calculate the average forest area for each country
data['Average'] = data.iloc[:, 5:].mean(axis=1)

# Display the country names and their corresponding average forest area
print(data[['Country Name', 'Average']])

# Let's say you want to extract data for 'Aruba'
country_name = 'Aruba'

# Filter the data for the specified country
country_data = data[data['Country Name'] == country_name]

# Display the extracted data for the country
print(country_data)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Aruba's forest area data
years = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020])  # Years from the dataset
forest_area = np.array([np.nan, np.nan, np.nan, np.nan, 4.2, 4.2, 4.2])  # Forest area values for Aruba

# Define a linear function
def linear_model(x, m, c):
    return m * x + c

# Fit the linear model to the data
params, _ = curve_fit(linear_model, years[4:], forest_area[4:])  # Considering data from 2000 onwards

# Get the parameters (slope and intercept)
slope, intercept = params

# Predicting for future years
future_years = np.array([2025, 2030])  # Replace this with the years you want to predict for
predicted_forest_area = linear_model(future_years, slope, intercept)

# Plotting the data and the fitted line
plt.scatter(years, forest_area, label='Actual Data')
plt.plot(years, linear_model(years, slope, intercept), color='red', label='Fitted Line')
plt.scatter(future_years, predicted_forest_area, color='green', label='Predicted Data')
plt.xlabel('Year')
plt.ylabel('Forest Area (sq. km)')
plt.title('Forest Area Prediction for Aruba')
plt.legend()
plt.show()



import pandas as pd

# Replace the file path with your actual file location
file_path = r"API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path, skiprows=4)  # Skip the first 4 rows as they contain metadata

# Display the first few rows of the DataFrame
print(data.head())

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = "API_AG.LND.FRST.K2_DS2_en_csv_v2_6302271.csv"
data = pd.read_csv(file_path, skiprows=4)  # Skip first 4 rows with metadata

# Drop unnecessary columns and rows
data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'], inplace=True)
data.drop([0, 1], inplace=True)  # Drop first two rows as they contain non-numeric data

# Set 'Country Name' as the index
data.set_index('Country Name', inplace=True)

# Remove columns containing non-numeric data
data = data.apply(pd.to_numeric, errors='coerce')

# Drop columns with all NaN values
data.dropna(axis=1, how='all', inplace=True)

# Transpose the data for easier plotting
data_transposed = data.T

# Convert index to datetime
data_transposed.index = pd.to_datetime(data_transposed.index, errors='coerce')

# Plotting
plt.figure(figsize=(10, 6))
for country in data_transposed.columns:
    plt.plot(data_transposed.index, data_transposed[country], label=country)

plt.title('Forest Area Over Time')
plt.xlabel('Year')
plt.ylabel('Forest Area (sq. km)')
plt.legend()
plt.grid(True)
plt.show()


top_10_countries = data.iloc[:, :-1].sum().nlargest(10).index
data_top_10 = data[top_10_countries]

plt.figure(figsize=(10, 6))
for country in top_10_countries:
    plt.fill_between(data_top_10.index, data_top_10[country], label=country, alpha=0.5)

plt.title('Forest Area Over Time (Top 10 Countries)')
plt.xlabel('Year')
plt.ylabel('Forest Area (sq. km)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=95)

plt.show()


plt.figure(figsize=(10, 6))
data_top_10.sum().plot(kind='bar')
plt.title('Total Forest Area by Top 10 Countries')
plt.xlabel('Country')
plt.ylabel('Total Forest Area (sq. km)')
plt.grid(axis='y')
plt.show()

top_10_countries = data.iloc[:, :-1].sum().nlargest(10).index
data_top_10 = data[top_10_countries]

plt.figure(figsize=(10, 6))
data_top_10.boxplot()
plt.title('Distribution of Forest Area by Top 10 Countries')
plt.xlabel('Countries')
plt.ylabel('Forest Area (sq. km)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

top_10_countries = data.iloc[:, :-1].sum().nlargest(10).index
data_top_10 = data[top_10_countries]

plt.figure(figsize=(10, 6))
plt.violinplot(data_top_10.values.T)
plt.title('Distribution of Forest Area by Top 10 Countries')
plt.xlabel('Countries')
plt.ylabel('Forest Area (sq. km)')
plt.xticks(ticks=range(1, len(top_10_countries) + 1), labels=top_10_countries, rotation=45)
plt.grid(axis='y')
plt.show()

import seaborn as sns
top_10_countries = data.iloc[:, :-1].sum().nlargest(10).index
data_top_10 = data[top_10_countries]

plt.figure(figsize=(12, 8))
sns.heatmap(data_top_10.T, cmap='YlGnBu')
plt.title('Forest Area Heatmap (Top 10 Countries)')
plt.xlabel('Year')
plt.ylabel('Country')
plt.show()
