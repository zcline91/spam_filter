import email
from email.policy import default
import csv
import logging
from collections import Counter


from ..emailextract import (ACCEPTED_CHARSETS, extract_email_data, 
    EmailEncodingError)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CorpusDirectoryStructureError(Exception):
    pass


class BaseCorpusExtractor:
    """An object for extracting email data from a corpus. 
    This class serves as a base class to be inherited from
    for any particular corpus, and should not be instatiated
    itself. Subclasses should implement get_index() and 
    redefine expected_directory_structure as a nested dict."""


    # Set a baseline list of accepted charsets. Subclasses can either expand 
    # this or replace it as need be.
    accepted_charsets = ACCEPTED_CHARSETS

    # The base class doesn't assume any directory structure. For subclasses, 
    # redefine this as an appropriate nested dict/list where each folder or
    # file is represented by a dict with keys "name" and "type" (directory or 
    # file) at minimum, and "children" if type==directory. In that case, the 
    # value of children is another list with the same structure
    expected_directory_structure = []


    def __init__(self, root_path, email_list=None):
        """The argument `email_list` should be a list of
        two-tuples of the form (int, str), where the first value is 
        0 for ham and 1 for spam, and the second is the path of an 
        email relative to root_path (a Path object). If email_list 
        is not passed, one will be created using 
        self.create_email_list()."""
        self.root_path = root_path
        self.verify_directory_structure()
        self.email_list = email_list 
        if email_list is None:
            self.create_email_list()
        else:
            self.email_list = email_list
    
    def verify_directory_structure(self):
        """Check that self.root_path satisfies the expected directory
        structure."""
        def check_contents(dirpath, children):
            """Check that the children of the directory at dirpath has 
            the appropriate structure, recursively."""
            for child in children:
                childpath = (dirpath / child["name"])
                if not childpath.exists():
                    raise CorpusDirectoryStructureError(
                        f"{childpath} is not present")
                if child["type"] == "file":
                    if not childpath.is_file():
                        raise CorpusDirectoryStructureError(
                            f"{childpath} expected to be of type file, "
                            "but it is not")
                elif child["type"] == "directory":
                    if not childpath.is_dir():
                        raise CorpusDirectoryStructureError(
                            f"{childpath} expected to be of type directory, "
                            "but it is not")
                    if "children" in child:
                        check_contents((dirpath / child["name"]), 
                            child["children"])
                else:
                    raise CorpusDirectoryStructureError(
                        f"Unknown type {child['type']} for {childpath}")
        check_contents(self.root_path, self.expected_directory_structure)
        return True

    def get_index(self):
        """The logic for this method will change based on the subclass.
        It should be implemented to return a pandas dataframe with an 
        integer first column "spam" (1 for spam, 0 for ham) and a string
        second column "path" giving the path of an email file relative 
        to self.root_path.""" 
        pass

    def create_email_list(self):
        """Sets self.email_list to a list of tuples of the form (int, str)
        where the first value is 0 for ham and 1 for spam, and the second 
        is the path of an email relative to self.root_path."""
        logger.debug(
            f"Gathering the list of emails and types in {self.root_path}")
        index = self.get_index()
        self.email_list = list(
            index.itertuples(index=False, name=None)
        )

    def create_csv(self, output_path):
        """Process the emails in `self.root_path` into a CSV file 
        `output_path`."""
        if not output_path.parent.exists():
            raise FileNotFoundError(f"{output_path.parent} does not exist."
                "Be sure 'output_path' is an existing directory.")
        logger.info(f"Extracting emails from {self.root_path} into "
            f"{output_path}")
        logger.info(f"Opening/reading all email files in {self.root_path} "
            "This may take awhile...")
        contents = []
        # Add counters for error types and rejected charsets
        error_counter = Counter({"missing": 0, "encoding": 0})
        rejected_charset_counter = Counter()
        spam_encode = {1: "spam", 0: "ham"}

        for email_type, relpath in self.email_list:
            # Try to open the email, extract info from it, and add the info as
            # row data for the CSV
            filepath = self.root_path / relpath
            logger.debug(f"Trying to extract {spam_encode[email_type]} email "
                        f"at {filepath}")
            try:
                email_obj = email.message_from_bytes(filepath.read_bytes(),
                                                    policy=default)
            except (FileNotFoundError, OSError) as e:
                error_counter["missing"] += 1
                logger.exception(e)
                continue
            try:
                email_data = extract_email_data(email_obj, 
                    accepted_charsets=self.accepted_charsets)
            except EmailEncodingError as e:
                error_counter["encoding"] += 1
                if "Unacceptable charset" in str(e):
                    charset = str(e).split()[-1]
                    rejected_charset_counter[charset] += 1
                logger.debug(f"Email at {filepath} could not be extracted: "
                    f"{e}")
                continue
            row = [relpath, email_type, *email_data]
            contents.append(row)
        sorted_rejected_charset_counts = sorted(
            rejected_charset_counter.items(),
            key=lambda x: x[1], reverse=True
        )
        logger.info(f"{len(contents)} out of {len(self.email_list)} emails "
            f"extracted. ({error_counter['encoding']} were rejected for "
            f"encoding reasons, {error_counter['missing']} referenced files "
            f"not found in the root path)\nMost common rejected charsets:\n"
            + "\n".join(
                f"\t{charset}: {count}" 
                for charset, count in sorted_rejected_charset_counts[:10]
            )
        )
        column_names = ('path', 'spam', 'subject', 'body')
        with output_path.open("wt", encoding="utf-8") as output_file:
            outputwriter = csv.writer(output_file, dialect='unix', escapechar="\\")
            outputwriter.writerow(column_names)
            outputwriter.writerows(contents)
