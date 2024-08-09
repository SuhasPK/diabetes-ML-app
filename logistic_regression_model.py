import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
file_path = "data/diabetes_data_upload_clean.csv"  # Update the path if necessary
df = pd.read_csv(file_path)

# Check for missing values
print("Missing values in the dataset:")
print(df.isna().sum())

# Drop rows where the target variable 'class' is NaN
df.dropna(subset=['class'], inplace=True)

# Define mapping dictionaries
label_dict = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}
target_label_map = {"Negative": 0, "Positive": 1}

# Print initial shape and sample data
print(f"Initial shape of data: {df.shape}")
print("Initial sample data:")
print(df.head())

# Apply mapping and check changes
df['gender'] = df['gender'].map(gender_map)
df['polyuria'] = df['polyuria'].map(label_dict)
df['polydipsia'] = df['polydipsia'].map(label_dict)
df['sudden_weight_loss'] = df['sudden_weight_loss'].map(label_dict)
df['weakness'] = df['weakness'].map(label_dict)
df['polyphagia'] = df['polyphagia'].map(label_dict)
df['genital_thrush'] = df['genital_thrush'].map(label_dict)
df['visual_blurring'] = df['visual_blurring'].map(label_dict)
df['itching'] = df['itching'].map(label_dict)
df['irritability'] = df['irritability'].map(label_dict)
df['delayed_healing'] = df['delayed_healing'].map(label_dict)
df['partial_paresis'] = df['partial_paresis'].map(label_dict)
df['muscle_stiffness'] = df['muscle_stiffness'].map(label_dict)
df['alopecia'] = df['alopecia'].map(label_dict)
df['obesity'] = df['obesity'].map(label_dict)
df['class'] = df['class'].map(target_label_map)

# Check for missing values and shape after mapping
print("Missing values after mapping:")
print(df.isna().sum())

# Ensure that there are no NaN values in the dataset
df.dropna(inplace=True)

# Check the shape of the dataset after dropping NaNs
print(f"Data shape after dropping NaNs: {df.shape}")

# Splitting the dataset into features (X) and target (y)
X = df.drop(columns='class')
y = df['class']

# Print shapes and sample data
print(f"Target variable 'y' shape: {y.shape}")
print(f"Features 'X' shape: {X.shape}")
print("Sample features data:")
print(X.head())

# Ensure there's data to split
if X.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
else:
    print("Error: No samples left for splitting.")
    exit()

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model as a .pkl file
model_filename = "data/logistic_regression_model_09_08_2024.pkl"
joblib.dump(model, model_filename)

print(f"Model saved as {model_filename}")
