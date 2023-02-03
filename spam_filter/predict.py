import email
from email.policy import default
import joblib

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


from data_processing.emailextract import extract_email_data


def create_email(email_file):
    return email.message_from_binary_file(email_file, policy=default)

def extract_email(email_obj):
    sub, body = extract_email_data(email_obj)
    return pd.DataFrame({'body': [body], 'subject': [sub]})

    
email_preparation_pipeline = Pipeline([
    ('email_obj',  FunctionTransformer(create_email)),
    ('email_body_sub', FunctionTransformer(extract_email)),
])

clf = joblib.load('model.joblib')

pred_pipeline = Pipeline([
    ('email_prep', email_preparation_pipeline),
    ('classifier', clf),
])

joblib.dump(pred_pipeline, 'predictor.joblib')
