from flask import Flask, render_template, request

import pandas as pd
import joblib


model = joblib.load('text_model.joblib')

app = Flask(__name__)


@app.route("/",methods=["GET"])
def landing_page():
    """
    Renders landing page
    """
    return render_template("landing.html")

@app.route("/",methods=["POST"])
def get_result():
    return handle_text_request(request)


def retrieve_probas(email_df):
    probs = model.predict_proba(email_df)[0]
    output_str = f"Probabilities: {probs[0]*100:.2f}% likely to be ham and {probs[1]*100:.2f}% likely to be spam."
    return output_str
    

def handle_text_request(request):
    """
    Renders an input form for GET requests and displays results for the given
    posted question for a POST request
    :param request: http request
    :return: Render an input form or results depending on request type
    """
    subject = str(request.form.get("subject"))
    print(subject)
    body = str(request.form.get("body"))
    email_df = pd.DataFrame({'subject': [subject,],'body': [body,]})
    probabilities = retrieve_probas(email_df)
    payload = {
        "input_subject": subject,   
        "input_body": body,
        "probabilities": probabilities
    }
    return render_template("results.html", **payload)
