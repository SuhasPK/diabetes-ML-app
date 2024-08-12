import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/diabetes_data_upload_clean.csv')  # Adjust path if needed
X = data.drop('class', axis=1)  # Use 'class' as target column based on your dataset
y = data['class']  # Adjust according to your dataset

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model and scaler
joblib.dump(model, 'models/logistic_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
