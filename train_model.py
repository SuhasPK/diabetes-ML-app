import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/diabetes_data_upload_clean.csv')  # Adjust path if needed
X = data.drop('class', axis=1)  # Use 'class' as target column based on your dataset
y = data['class']  # Adjust according to your dataset

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = LogisticRegression()

# Define hyperparameters to tune
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Norm used in the penalization
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']  # Algorithm to use
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Hyperparameters: {best_params}")
print(f"Best Training Accuracy: {grid_search.best_score_}")

# Evaluate on the test set
y_pred = best_model.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the best model and scaler
joblib.dump(best_model, 'models/best_logistic_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
