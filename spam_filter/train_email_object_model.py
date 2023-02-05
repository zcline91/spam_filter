import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_processing import email_to_df


clf = joblib.load('model.joblib')

pred_pipeline = Pipeline([
    ('email_prep', FunctionTransformer(email_to_df)),
    ('classifier', clf),
])

joblib.dump(pred_pipeline, 'object_model.joblib')
