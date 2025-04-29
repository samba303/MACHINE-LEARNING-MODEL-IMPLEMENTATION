# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# Replace 'your_dataset.csv' with your actual dataset file.
data = pd.read_csv('your_dataset.csv')

# Preprocess the data
# Replace 'target_column' with the actual target column name.
X = data.drop(columns=['target_column'])
y = data['target_column']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM model
model = SVC(kernel='linear', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, 'svm_predictive_model.pkl')
print("Model saved as 'svm_predictive_model.pkl'"
