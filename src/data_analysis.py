
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/archive/location_aware_gis_leakage_dataset_with_units.csv")
# Display first five rows
print("FIRST FIVE ROWS")
print(df.head())

# Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Dataset information
print("\nDATASET INFORMATION")
df.info()

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check duplicate rows
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Check leakage distribution
print("\nLEAKAGE DISTRIBUTION")
print(df["Leakage_Flag"].value_counts())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())
