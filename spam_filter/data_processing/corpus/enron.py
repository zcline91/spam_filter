import logging
from pathlib import Path

import pandas as pd

from .corpus_extractor import BaseCorpusExtractor
from ..emailextract import ACCEPTED_CHARSETS


# Expected enron_root_path/ structure:
# enron_root_path/
# ├─raw/
#  ├─ beck-s/
#  ├─ BG/
#  ├─ farmer-d/
#  ├─ GP/
#  ├─ kaminski-v/
#  ├─ kitchen-l/
#  ├─ lokay-m/
#  ├─ SH/
#  ├─ williams-w3/
#
# If you extract each of the "raw" tarballs into a common directory,
# this should be the result.


logger = logging.getLogger(__name__)

__location__ = Path.cwd().joinpath(Path(__file__).parent).resolve()


class EnronCorpusExtractor(BaseCorpusExtractor):

    accepted_charsets = ACCEPTED_CHARSETS + ('iso-8859-7',)

    expected_directory_structure = [
        {"name": "raw", "type": "directory", "children": [
            {"name": "beck-s", "type": "directory"},
            {"name": "BG", "type": "directory"},
            {"name": "farmer-d", "type": "directory"},
            {"name": "GP", "type": "directory"},
            {"name": "kaminski-v", "type": "directory"},
            {"name": "kitchen-l", "type": "directory"},
            {"name": "lokay-m", "type": "directory"},
            {"name": "SH", "type": "directory"},
            {"name": "williams-w3", "type": "directory"},
        ]}
    ]

    def get_index(self):
        index_path = __location__ / "_enron_index.csv"
        logger.info(f"Fetching index file from {index_path}")
        return pd.read_csv(index_path, sep=',')
