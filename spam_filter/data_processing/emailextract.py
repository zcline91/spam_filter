import logging 


from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


ACCEPTED_CHARSETS = ('', 'iso-8859-1', 'us-ascii', 'iso-646-us', 'us-ascii',
                     'utf-8', 'windows-1252', 'ansi_x3.4-1968')
# The inclusion of the empty string will result in emails being included for
# which no charset is found. It seems like most of these can still be parsed,
# and since so many emails in the dataset had no charset specified, it seemed
# sensible to include. If we can suffer the loss of size in the training set
# for a more predictably formatted individuals, we could eliminate the emptry
# string here.


class EmailContentTypeError(Exception):
    pass


class EmailEncodingError(Exception):
    pass


def get_email_text(email_obj, accepted_charsets=ACCEPTED_CHARSETS):
    """Returns the text contents of the email object.
    - It is assumed the email has content_type 'text/plain' or 'text/html'.
      (HTML emails are parsed using BeautifulSoup.) If not, an 
      EmailContentTypeError exception is raised.
    - The optional argument `accepted_charsets` should be an iterable of 
      string representations of charsets (e.g. 'utf-8'). If the email_obj has a 
      charset not in this list, an EmailEncodingError exception is raised. 
    - If the parser from the email package is unable to parse the text, an
      EmailEncodingError exception is raised."""
    # Verify the email_obj is "text/plain" or "text/html" type
    contenttype = email_obj.get_content_type()
    if contenttype not in ('text/plain', 'text/html'):
        raise EmailContentTypeError(f'{email_obj!r} does not have type '
                                    '"text/plain" or "text/html". It has '
                                    f'type "{contenttype}".')
    # Check the charset.
    charset = email_obj.get_content_charset(failobj='')
    if charset == '':
        logger.debug(f"No charset was found for {email_obj!r}. Setting to ''")
    if charset not in accepted_charsets:
        raise EmailEncodingError(f"Unacceptable charset {charset}")
    # Get the contents and parse based on the subtype.
    try:
        content = email_obj.get_content()
    except LookupError as e:
        logger.error(e)
        raise EmailEncodingError from e
    if contenttype == 'text/plain':
        return content
    elif contenttype == 'text/html':
        return BeautifulSoup(content, features="html.parser").get_text()


def extract_email_data(email_obj, accepted_charsets=ACCEPTED_CHARSETS):
    """Returns tuple of the email's subject line and text contents.
    Neither is prevented from being the empty string. Only parts of the email 
    of type 'text/plain' or 'text/html' (and with charset in 
    `accepted_charsets`) are parsed for textual contents. Every other part of
    the email is skipped. 
    NOTE: We can alter this function if we decide 
    - we would like more data from each email, e.g.
        - the number of recipients
        - whether or not there is (a particular filetype of) an attachment 
    - we would like to include some sort of textual representation for 
      other content types, e.g.
        - for a part of type 'img/jpeg', writing '\\nIMAGE\\n' rather than '' 
          to the contents field"""
    subject = email_obj['subject']
    content = ''
    for part in email_obj.walk():
        if part.is_multipart():
            continue
        contenttype = part.get_content_type()
        if contenttype in ('text/plain', 'text/html'):
            try:
                content += get_email_text(part, 
                                          accepted_charsets=accepted_charsets)
            except EmailEncodingError:
                raise
        else:
            logger.debug(f"Skipping email part of type {contenttype}")
    return (subject, content)