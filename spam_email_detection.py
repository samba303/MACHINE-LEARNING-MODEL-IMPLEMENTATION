import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
# Replace 'spam_dataset.csv' with the path to your dataset
# The dataset should have two columns: 'text' (email content) and 'label' ('spam' or 'not spam')
df = pd.read_csv('spam_dataset.csv')

# Preprocess the data
# Convert labels to binary (1 for spam, 0 for not spam)
df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split dataset into features and labels
X = df['text']
y = df['label']

# Convert text data to numerical data using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the predictive model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test the model with a custom email
sample_email = ["Congratulations! You've won a free iPhone. Click here to claim your prize."]
sample_email_vectorized = vectorizer.transform(sample_email)
prediction = model.predict(sample_email_vectorized)
print("\nPrediction for sample email:", "Spam" if prediction[0] == 1 else "Not Spam")