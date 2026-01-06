# Step 1: Import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

# TASK 1
# Step 2: Load the dataset
df = pd.read_csv('dataset.csv', encoding='UTF-8-SIG')

# Step 3: Print the total number of rows
print("Total number of rows:", len(df))

# Step 4: Print the top 5 rows to inspect the data
print("\nTop 5 rows of the dataset:")
print(df.head())

# Step 5 (optional): Display basic info about data types and structure
print("\nDataset Info:")
print(df.info())

# TASK 2

# Define features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Confirm shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# TASK 3