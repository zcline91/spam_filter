from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .email_cleaning import normalize_spaces, \
    all_whitespace_to_na, drop_na_both, drop_duplicates, \
    drop_multipart_messages, drop_non_english


# normalizes a dataframe of email objects (with a 'subject' and 'body' field)
email_cleaning = Pipeline([
    ('space_normalizer', FunctionTransformer(normalize_spaces)),
    ('empty_nullify', FunctionTransformer(all_whitespace_to_na)),
])


# normalizes and then prepares a dataframe of email objects from a corpus 
# or multiple corpora 
# For instance, duplicates will be dropped.
# The last transformer is necessary because some email messages were not 
# properly picked up as multipart by the email package, causing the body 
# to contain non-text data encoded as text. The easiest solution is to drop 
# the few emails that got by.
corpus_prep = Pipeline([
    ('email_cleaner', email_cleaning),
    ('non_english_dropper', FunctionTransformer(drop_non_english)),
    ('empty_dropper', FunctionTransformer(drop_na_both)),
    ('duplicate_dropper', FunctionTransformer(drop_duplicates)),
    ('multipart_dropper', FunctionTransformer(drop_multipart_messages))
])
