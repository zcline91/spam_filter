import joblib
from sklearn.metrics import fbeta_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer, Normalizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier



from data_processing.preprocessing import load_train_test_classes, \
    load_train_test_docs
from data_processing.spacy import Lemmatizer
from data_processing.spacy.dochandling import DocCreatorB


# Load the data
train_classes, test_classes = load_train_test_classes()
train_set, test_set = load_train_test_docs(train_classes, test_classes)
y_train = train_classes.to_numpy(dtype='int')
y_test = test_classes.to_numpy(dtype='int')


body_bow_pipeline = Pipeline([
    ('lemmas', Lemmatizer(del_stop=False, del_punct=False, del_num=False)),
    ('dict', DictVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)),
])
subject_bow_pipeline = Pipeline([
    ('lemmas', Lemmatizer(del_stop=False, del_punct=False, del_num=True)),
    ('dict', DictVectorizer()),
    ('bin', Binarizer()),
    ('norm', Normalizer()),
])

fit_clf = Pipeline([
    ('feature_eng', ColumnTransformer([
            ('body_bow', body_bow_pipeline, 1),
            ('subject_bow', subject_bow_pipeline, 0),
        ])),
    ('sgd', SGDClassifier(loss='modified_huber', 
        class_weight={0: .85, 1: .15}, alpha=2e-5,
        random_state=42))
])

print("Training model...")
fit_clf.fit(train_set, y_train)

print("Testing model...")
y_test_predict = fit_clf.predict(test_set)

print(classification_report(y_test, y_test_predict, 
    target_names=['ham', 'spam'], digits=5))

print(f"F_half score: {fbeta_score(y_test, y_test_predict, beta=.5)}")

clf = Pipeline([
    ('create_docs', DocCreator()),
    ('fit_clf', fit_clf),
])
joblib.dump(clf, 'model.joblib')
