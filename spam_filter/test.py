import joblib
from pathlib import Path
import email
from email.policy import default


model = joblib.load('predictor.joblib')

filepath = Path('test.eml')

def create_email(email_file):
    return email.message_from_binary_file(email_file, policy=default)


with open(filepath, 'rb') as email_file:
    email_obj = create_email(email_file)

print(model.predict_proba([email_obj,]))
