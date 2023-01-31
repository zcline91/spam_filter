import logging
import re

import numpy as np
from langdetect import detect, LangDetectException, DetectorFactory


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DetectorFactory.seed = 0


def normalize_spaces(email_df):
    """Return a dataframe with all whitespace in the 'subject'
    and 'body' fields of the original replaced with a single
    blank: ' '"""
    logger.info("Normalizing spaces in email dataframe's subject and body "
        "fields")
    pat = re.compile(r'\s+')
    return email_df.assign(
        subject=email_df['subject'].str.replace(pat, ' ', regex=True),
        body=email_df['body'].str.replace(pat, ' ', regex=True)
    )


def all_whitespace_to_na(email_df):
    """Return a dataframe in which any all-whitespace 'subject'
    or 'body' fields of the original have been nullified"""
    logger.info("Setting all-whitespace body and subject fields of "
        "email dataframe to None.")
    pat = re.compile(r'^\s*$')
    return email_df.assign(
        subject=(email_df['subject']
            .str.replace(pat, '', regex=True) # Intermediate step for speed
            .replace('', np.nan)),
        body=(email_df['body']
            .str.replace(pat, '', regex=True)
            .replace('', np.nan))
    )


def drop_na_both(email_df):
    """Return a dataframe in which any rows where both 'subject'
    and 'body' are null have been dropped"""
    logger.info(f"Dropping any emails from email dataframe for which both "
        "subject and body are None")
    return email_df.dropna(subset=['subject', 'body'])


def drop_non_english(email_df):
    """Return a dataframe where all rows with non-English email bodies
    have been dropped."""
    logger.info("Dropping any emails in non-English languages. "
        "This may take a while...")
    def english(body_text):
        try:
            return detect(body_text) == 'en'
        except LangDetectException:
            return False
        except TypeError:
            return False
    return email_df[email_df['body'].map(english)]


def drop_duplicates(email_df):
    """Return a dataframe with no duplicate rows (checking 
    only the equality of 'subject' and 'body')"""
    logger.info("Dropping any duplicate emails from dataframe (same subject "
        "and body)")
    return email_df.drop_duplicates(['subject', 'body'])


def drop_multipart_messages(email_df):
    """Return a dataframe in which any rows where 'body' contains
    the strings 'multipart' or 'Multipart' have been dropped."""
    logger.info(f"Dropping any messages from email dataframe that contain the "
        "string '[M|m]ultipart' in their body.")
    return email_df[~email_df['body'].str.contains('multipart') 
        & ~email_df['body'].str.contains('Multipart')]
