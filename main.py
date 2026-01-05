# Step 1: Import pandas
import pandas as pd
import chardet

# Read first few bytes to detect encoding
with open('dataset.csv', 'rb') as f:
    rawdata = f.read(10000)  # read first 10 KB
    result = chardet.detect(rawdata)
    print(result)

# Step 2: Load the dataset
# Replace 'your_dataset.csv' with the actual file path
df = pd.read_csv('dataset.csv', encoding=result['encoding'])

# Step 3: Print the total number of rows
print("Total number of rows:", len(df))

# Step 4: Print the top 5 rows to inspect the data
print("\nTop 5 rows of the dataset:")
print(df.head())

# Step 5 (optional): Display basic info about data types and structure
print("\nDataset Info:")
print(df.info())