import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import joblib

raw_emails = pd.read_csv('theemails - emails.csv')

email_data=raw_emails.where((pd.notnull(raw_emails)),'')
email_data.head()

X=email_data['text']
Y=email_data['spam']

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25, random_state=1234)

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english', lowercase=True)
X_train_feature=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

model=LogisticRegression()
model.fit(X_train_feature, Y_train)

training_test=model.predict(X_train_feature)
training_accuracy=accuracy_score(Y_train, training_test)
print('Accuracy:', training_accuracy)

testing_test=model.predict(X_test_features)
testing_accuracy=accuracy_score(Y_test,testing_test)
print('Accuracy:',testing_accuracy)


joblib.dump(model, 'spam_model.pkl')
joblib.dump(feature_extraction, 'vectorizer.pkl')

app = Flask(__name__)
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_spam(email_content):
  email_features=vectorizer.transform([email_content])
  prediction=model.predict(email_features)
  return "spam" if prediction[0] == 1 else "ham"

@app.route('/check_spam',methods=['POST'])
def check_spam():
  try:
    data=request.get_json()
    email_content=data.get('email')
    result=predict_spam(email_content)
    return jsonify({'result':result})
  except Exception as e:
    return jsonify({'error':str(e)})

@app.route('/')
def index():
  with open('index.html', 'r') as f:
    return f.read()

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))  # Default must match Flask's fallback
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
