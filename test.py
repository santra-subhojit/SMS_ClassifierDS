# train_save_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump

# Sample training data
X_train = [
    "Congratulations! You've won a free iPhone. Click here to claim your prize now.",
    "Limited time offer! Get cheap loans at 0% interest. Apply now!",
    "Urgent: Update your account information immediately.",
    "Hey, are we still on for lunch tomorrow?",
    "Don't forget to submit the report by EOD.",
    "Reminder: Your meeting is scheduled for 3 PM today."
]
y_train = [1, 1, 1, 0, 0, 0]  # Example labels (1 for spam, 0 for not spam)

# Create a pipeline that includes the vectorizer and the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train (fit) the model on the training data
model.fit(X_train, y_train)

# Save the trained model
dump(model, 'model.pkl')
print("Model trained and saved as 'model.pkl'.")
