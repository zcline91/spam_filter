import pytest
import pandas as pd

from spam_filter.data_processing.preprocessing.email_cleaning import \
    normalize_spaces, all_whitespace_to_na, drop_na_both, drop_duplicates, \
    drop_multipart_messages, drop_non_english


def test_drop_non_english():
    lang_df = pd.DataFrame({
        'subject': ['foo']*7,
        'body': ['Hello World!', 'ওহে বিশ্ব!', 'Hallo Welt!', 'Hola món!', 
            '¡Hola Mundo!', ' ', None]
    })
    dropped = drop_non_english(lang_df)
    assert dropped.equals(pd.DataFrame({
        'subject': ['foo',],
        'body': ['Hello World!',]
    }))


def test_normalize_spaces():
    var = ["foo bar", "foo  bar", "foo\tbar", "foo\nbar", 
        "foo\r\nbar", "foo\r\n\t bar"]
    variable_spaces = pd.DataFrame({"subject": var, "body": var})
    normalized = normalize_spaces(variable_spaces)
    assert len(normalized) == len(variable_spaces)
    assert len(normalized.drop_duplicates()) == 1


@pytest.fixture
def whitespaces():
    return pd.DataFrame({
        "subject": ["foo", "bar", "biz", "baz", "foo",  " ", "\n", "\t", \
            "\t \n", " ", "", None],
        "body": [" ", "\n", "\t", "\t \n", "foo", "bar", "biz", "baz", \
            "foo", " ", "", None]
    })


def test_all_whitespace_to_na(whitespaces):
    whitespace_to_na = all_whitespace_to_na(whitespaces)
    assert whitespace_to_na.equals(pd.DataFrame({
        "subject": ["foo", "bar", "biz", "baz", "foo",  None, None, None, \
            None, None, None, None],
        "body": [None, None, None, None, "foo", "bar", "biz", "baz", \
            "foo", None, None, None]
    }))


def test_drop_na_both(whitespaces):
    whitespace_drop_na_both = drop_na_both(whitespaces)
    assert whitespace_drop_na_both.equals(pd.DataFrame({
        "subject": ["foo", "bar", "biz", "baz", "foo",  " ", "\n", "\t", \
            "\t \n", " ", ""],
        "body": [" ", "\n", "\t", "\t \n", "foo", "bar", "biz", "baz", \
            "foo", " ", ""]
    }))


def test_drop_duplicates():
    dups = pd.DataFrame({
        "subject": ["foo",] * 3 + ["bar",] + ["baz",] * 2,
        "body": ["foo",] * 2 + ["bar",] * 2 + ["baz",] * 2 
    })
    dropped_dups = drop_duplicates(dups)
    assert dropped_dups.equals(pd.DataFrame(
        {
            "subject": ["foo", "foo", "bar", "baz"],
            "body": ["foo", "bar", "bar", "baz"]
        }, 
        index=[0,2,3,4]
    ))


def test_drop_multipart_messages():
    msgs = pd.DataFrame({
        "subject": ["foo"]*3,
        "body": ["bar", "bar multipart", "bar Multipart"]
    })
    dropped_multi = drop_multipart_messages(msgs)
    assert dropped_multi.equals(pd.DataFrame(
        {"subject": ["foo"], "body": ["bar"]}
    ))
