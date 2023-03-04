# Spam Filter

For tech enthusiasts and people with homelabs, one of the more daunting potential projects is setting up a self-hosted email server. Even after the server itself is set up and is successfully receiving and sending email, any email address on the domain will eventually end up on countless email lists and will start receiving a barage of spam email. This project provides a way to incorporate spam detection in such a setup.

We have developed a model for detecting spam with simple API calls that can be incorporated into the workflow. We currently have an instance of the API site running, and while we do not store the requests made to the API, you can self-host the spam detection API site itself for security or reliability. It runs in a Docker container.

For information about hosting or using the API site, visit the [API repo](https://github.com/zcline91/spam_filter_api).

You can also [see the spam filter in action here](https://mreeks91-spam-filter-site-spam-app-4f7240.streamlit.app/), where the predictions are being made on the backend using the deployed model via the API.

This repository contains the source to create the models used in the API (really one underlying classification model, but two finished models for accepting two different types of input).

## Reproducing the Model(s)

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

    The output of the first command, `text_model.joblib`, accepts dataframes or 2D arrays of (subject, body) emails.
    The output of the second command, `object_model.joblib`, accepts iterables of email objects (as created by the email package).

## Design and Development Process

### ETL / Preprocessing

After deciding to use machine learning to build a spam filter, we desired a large dataset to get the best performance possible, so we found 5 publicly available corpora of labelled email data (the datasets are linked in [the previous section](#reproducing-the-models)). Since these corpora did not all store their data in the same format, we needed to extract data in a consistent way for use in training.

While some of the metadata in the header of an email can be extremely useful for spam classification, not all of the corpora contained this metadata. Two options presented themselves: (1) limiting the dataset only to only those emails containing full headers, or (2) training our classifier purely on the text of the subject and body of the email. We decided on option (2) for this project.

#### Corpus Gathering and Extraction

The code in `data_processing/corpus` and `data_processing/email_extract` is for extracting the subject and body text contents from the various corpora of emails, and storing the results in a single CSV file (one for each corpus). This sometimes requires extracting text contents from an HTML-formatted email, for which BeautifulSoup was used.

The core functionality for data extraction was placed in a base class `BaseCorpusExtractor`, so that extractor classes for individual corpora only need to define a few class attributes and methods. **Because of this, it is very easy to add new corpora of emails to the dataset for later iterations.**

The CSVs contain the following columns:

- `path`: the relative path in the downloaded corpus to the email file (used for indexing instances in dataframes)
- `spam`: 1 for spam, 0 for ham
- `subject`: text contents of the subject
- `body`: text contents of the body

#### Preprocessing Text

Once we'd gathered the raw text of the subject and body of each email, we recognized that some emails appeared in multiple corpora (the corpora had overlapping raw sources themselves), and that not all the emails were written in English. While spam filtering is not an English-specific task, and obviously emails in other languages can be sent to English speakers, this was beyond the scope of our project.)

The code in `data_processing/preprocessing/email_cleaning.py` is for normalizing the space in emails, dropping non-English language emails, dropping empty emails (empty subject _and_ body), dropping duplicates, and finally, dropping any messages that errantly have content-type `multipart` (which would have been eliminated in [the previous step](#corpus-gathering-and-extraction) had the email followed MIME formatting correctly).

With the corpora email data properly preprocessed and stored, we split the data into training and test instances using hashes, with a test set ratio of approximately 20%.

Treating each training instance as two blocks of English text, we decided to use a pre-trained NLP model to extract data from the text. We ran the subject and body of each email through the spacy model `en_core_web_sm` (small Engligh model) to create spacy Doc objects, and stored them in .docbin files (serialized corpora of Doc objects that can loaded without rerunning the spacy model on the original text). These serialized files included storage of the index of each email (a MultiIndex consisting of the corpus name and path of the email in the corpus folder) to ensure that Docs loaded from the .docbin files are matched properly to their spam/ham classes. To ensure we could treat new email instances in the same way as our training and test instances, we created a custom scikit-learn Transformer class, DocCreator, to actually pass the text through the spacy model.

Finally, for easily loading the classes alongside the Doc objects, we created a set of smaller (filesize) CSV files that contained the MultiIndex of each email (corpus name, path) and its class.

### Feature Engineering

We decided to use a bag-of-words representation for our emails, but rather than using the verbatim tokens (words) in the email, we used the lemmas (e.g. `run` instead of `running`). The spacy model is able to recognize part of speech, singular/plural, etc. based not just on each individual token, but on its context. For extracting a dictionary of lemma counts from each subject and body, we created another custom scikit-learn Transformer class, Lemmatizer.

Both the subject and body were lemmatized, and then passed to a DictVectorizer to create vectors of lemma counts. At this point, the body and subject were treated slightly differently. We created an $\ell_2$-normalized tf-idf representation of the body, but an $\ell_2$-normalized binary representation of the subject. Essentially, rather than counting frequencies of words in the subject, we counted just whether or not a word occured at all.
At this point, we concatenated these features to form the final feature set for training.

### Training the model

With such a high-dimensional feature space (tens of thousands of lemmas), we thought it likely the data were close to linearly separable, and tried fitting a linear model with different algorithms such as Logistic Regression and Linear SVMs. Unsurprisingly, all of these performed with pretty high accuracy (> 90%) on a first attempt with no hyperparameter tuning. However, we decided to try fitting some non-linear models as well to see if performance could be improved.

For comparing models, we measured model quality with the $F_{\beta=\frac 1 2}$-score (weighting misclassified ham worse than misclassified spam). We fit models using Multinomial Naive Bayes, Decision Trees, Random Forests and Extra-Trees, boosting, and SVMs with kernel approximation. While these models performed similarly to the linear models, they took much longer to train.

With over $10^5$ samples in the training set, we chose stochastic gradient descent for training the linear models, allowing for hyperparameters of the model to be tuned quickly and efficiently through $N$-fold cross-validation.

Ultimately, after tuning the hyperparameters on the most promising linear and non-linear candidates, the best-performing model was a linear model trained using the modified Huber loss function. We did consider using a Voting classifier or other type of ensemble model, incorporating predictions from our best performing single models with different underlying assumptions, but the miniscule boost in performance did not justify the added training time or loss of explainability. The final model achieved an $F_{\frac 1 2}$-score of $.986$ on the test set. The classfication report is below:

```text
              precision    recall  f1-score   support

         ham    0.97407   0.99282   0.98336     17408
        spam    0.99085   0.96713   0.97885     13995

    accuracy                        0.98137     31403
   macro avg    0.98246   0.97998   0.98110     31403
weighted avg    0.98155   0.98137   0.98135     31403
```
