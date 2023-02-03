import email
from email.policy import default
import joblib
from pathlib import Path

import pandas as pd

from data_processing.emailextract import extract_email_data


def create_email(email_file):
    return email.message_from_binary_file(email_file, policy=default)

def extract_email(email_obj):
    body, sub = extract_email_data(email_obj)
    return pd.DataFrame({'body': [body], 'subject': [sub]})
    

model = joblib.load('predictor.joblib')

filepath = Path('test.eml')

with open(filepath, 'rb') as email_file:
    print(model.predict_proba(email_file))
