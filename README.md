# Set-Up

1. Create a virtual environment and install the dependencies in requirements.txt, e.g. using virtualenv:

    ```bash
    virtualenv .env
    source .env/bin/activate
    pip install -r requirements.txt
    ```

2. Download the following email corpora:
    - [NIST 2005 TREC Public Spam Corpus](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo)
    - [NIST 2006 TREC Public Spam Corpus](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo06)
    - [NIST 2007 TREC Public Spam Corpus](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo07)
    - [Ling-Spam](http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz)
    - [Enron-Spam](https://www2.aueb.gr/users/ion/data/enron-spam/)
    
    Unpack each of the tarballs for all the corpora and store them all in a single directory as such:

    ```text
    corpora/
    ├─ enron/
    ├─ lingspam_public/
    ├─ trec05p-1/
    ├─ trec06p/
    ├─ trec07p/
    ```

3. Edit the variables `CORPORA_CSV_PATH`, `DOCBIN_PATH`, and `SPAM_CLASS_PATH` in spam_filter/data_processing/settings.py to point to (existing) directories where you would like the compiled corpora, docbins, and class files stored. You can also edit the names that these files will be given if you'd like.
4. Extract the body and subject of the emails from the corpora directories into single .csv files (one for each corpus):

    ```bash
    python3 extract_from_corpus.py --all <corpus_path>
    ```

    where `<corpus_path>` is the path containing all the corpora files. This will take some time. (Run `extract_from_corpus.py -h` to see options, such as for only extracting from only one of the corpora at a time.) \
    NOTE: Since some of the spam emails in the corpora contain viruses, you will likely need to disable any form of realtime antivirus threat detection on your computer for this step, as otherwise, your antivirus software will delete some of the files. Remember to immediately turn it back on once this step is done. After this step, only the text contents of the subject and body of the email are needed, which are now stored in the created csv files at the path you set previously.
5. Run the small English Spacy pipeline on the subject and body of each email, extracted above, and store the results in spacy docbins:

    ```bash
    python3 create_docbins.py
    ```

    This will also take some time. If your computer doesn't have a lot of memory (<24GB), consider adding the option `--batchsize 1000` or `--batchsize 500` so less resources are spent on each batch before reloading the pipeline fresh. (Run `create_docbins.py -h` to see more options.)
6. Create csv files containing the classes (labels) of each email. This is stored as a boolean with True for spam and False for ham.
    ```bash
    python3 create_classes.py
    ```
7. Create the models:
    ```bash
    python3 train_text_model.py
    python3 train_email_object_model.py
    ```
    The output of the first command, `text_model.joblib` accepts dataframes or 2D arrays of (subject, body) emails.
    The output of the second command, `object_model.joblib` accepts iterables of email objects (as created by the email package).

# API Endpoints

## `/api/predict/text` 

### Input

```jsonc
{
    'instances': [{'subject': <str>, 'body': <str>},]
    'return_prob': <boolean>, // default: true
    'return_inputs': <boolean> // default: false
}
```

### Output

```jsonc
[
    {
        'prediction': <boolean>,
        'spam_probability': <number>, // optional
        'subject_head': <str>, // optional
        'body_head': <str>, //optional
    },
]
```