import logging
from pathlib import Path

import pandas as pd

from .corpus_extractor import BaseCorpusExtractor


# Expected self.root_path/ structure:
# self.root_path/
# ├─ bare/
# │  ├─ part1/
# │  ├─ part2/
# │  ├─ ...
# │  ├─ part10/
# ├─ lemm/
# ├─ lemm_stop/
# ├─ stop/
#
# (Right now, only `bare` is necessary.)
# This should be the structure of the directory 'lingspam_public'
# after extracting the downloaded tar file.


logger = logging.getLogger(__name__)

__location__ = Path.cwd().joinpath(Path(__file__).parent).resolve()


class LingCorpusExtractor(BaseCorpusExtractor):

    expected_directory_structure = [
        {"name": "bare", "type": "directory", "children": [
            {"name": "part1", "type": "directory"},
            {"name": "part2", "type": "directory"},
            {"name": "part3", "type": "directory"},
            {"name": "part4", "type": "directory"},
            {"name": "part5", "type": "directory"},
            {"name": "part6", "type": "directory"},
            {"name": "part7", "type": "directory"},
            {"name": "part8", "type": "directory"},
            {"name": "part9", "type": "directory"},
            {"name": "part10", "type": "directory"},
        ]}
    ]

    def get_index(self):
        index_path = __location__ / "_ling_index.csv"
        logger.info(f"Fetching index file from {index_path}")
        return pd.read_csv(index_path, sep=',')
