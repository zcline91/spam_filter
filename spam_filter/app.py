import email
from email.policy import default
import tempfile
from zipfile import ZipFile
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, \
    make_response, abort


ALLOWED_EXTENSIONS = {'eml'}
ALLOWED_ARCHIVES = {'zip'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.')[-1].lower() in ALLOWED_EXTENSIONS

def allowed_archive(filename):
    return '.' in filename and \
        filename.rsplit('.')[-1].lower() in ALLOWED_ARCHIVES


text_model = joblib.load('text_model.joblib')
object_model = joblib.load('object_model.joblib')

app = Flask(__name__)


@app.get("/")
def landing_page():
    return render_template("landing.html")


@app.get("/text")
def text_page():
    return render_template("text.html")


@app.get("/file")
def file_page():
    return render_template("file.html")

@app.post("/text")
def text_request():
    subject = request.form.get("subject")
    body = request.form.get("body")
    email_df = pd.DataFrame({'subject': [subject,],'body': [body,]})
    response_df = handle_request(email_df, 'text', return_prob=True)
    resp = make_response(redirect(url_for('results')))
    resp.set_cookie('spam_prob', f"{100 * response_df.loc[0,'spam_probability']:.3f}")
    resp.set_cookie('ham_prob', f"{100 * (1 - response_df.loc[0,'spam_probability']):.3f}")
    return resp


@app.post("/file")
def file_request():
    file = request.files['file']
    if file and allowed_file(file.filename):
        email_obj = email.message_from_binary_file(file, policy=default)
        response_df = handle_request(email_obj, "file", return_prob=True)
    else:
        abort(400)
    resp = make_response(redirect(url_for('results')))
    resp.set_cookie('spam_prob', f"{100 * response_df.loc[0,'spam_probability']:.3f}")
    resp.set_cookie('ham_prob', f"{100 * (1 - response_df.loc[0,'spam_probability']):.3f}")
    return resp


@app.get("/result")
def results():
# def results(**kwargs):
    # payload = {key: request.cookies.get(key) for key in 
        # ['input_subject', 'input_body', 'probabilities']}
    # return render_template("results.html", **payload)
    spam_prob = request.cookies['spam_prob']
    ham_prob = request.cookies['ham_prob']
    return render_template("results.html", spam_prob=spam_prob, ham_prob=ham_prob)


@app.post("/api/predict/text")
def api_predict_text():
    input = request.json
    email_df = pd.DataFrame(input['instances'])
    rp = input.setdefault('return_prob', True)
    ri = input.setdefault('return_inputs', False)
    response_df = handle_request(email_df,'text', rp, ri)
    return response_df.to_dict(orient='records')


@app.post("/api/predict/file")
def api_predict_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        email_obj = email.message_from_binary_file(file, policy=default)
    response_df = handle_request(email_obj,'file')
    return response_df.to_dict(orient='records')

@app.post("/api/predict/archive")
def api_predict_archive():
    archive = request.files['archive']
    frames = []
    if archive and allowed_archive(archive.filename):
        td = tempfile.TemporaryDirectory()
        zip_ref = ZipFile(archive)
        zip_ref.extractall(td.name)
        for filename in os.scandir(td.name):
            with open(filename.path, 'rb') as eml:
                email_obj = email.message_from_binary_file(eml, policy=default)
                frames.append(handle_request(email_obj, "file", return_prob=True).to_dict(orient='records')) #in future, pass list to handlerequests and receive 1 df
    return(frames)

def handle_request(email_input, model_type, return_prob=True, return_inputs=False):
    if model_type == "text":
        model = text_model
        response_df = pd.DataFrame({"pred": model.predict(email_input).astype(bool)})
        if return_prob:
            response_df['spam_probability'] = model.predict_proba(email_input)[:,1]
    elif model_type == "file":
        model = object_model
        response_df = pd.DataFrame({"pred": model.predict([email_input,]).astype(bool)})
        if return_prob:
            response_df['spam_probability'] = model.predict_proba([email_input,])[:,1]
    if return_inputs and model_type == "text":
        response_df['subject_head'] = email_input['subject'].str.slice(stop=100)
        response_df['body_head'] = email_input['body'].str.slice(stop=100)
    return response_df


def retrieve_probas(e, input_type):
    if input_type == "text":
        probs = text_model.predict_proba(e)[0]
        output_str = f"Probabilities: {probs[0]*100:.2f}% likely to be ham and {probs[1]*100:.2f}% likely to be spam."
    elif input_type == "file":
        email_obj = email.message_from_binary_file(e, policy=default)
        probs = object_model.predict_proba([email_obj,])[0]
        output_str = f"Probabilities: {probs[0]*100:.2f}% likely to be ham and {probs[1]*100:.2f}% likely to be spam."
    return output_str
