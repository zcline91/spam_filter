import logging

import pandas as pd

from .corpus_extractor import BaseCorpusExtractor


# Expected trec root_path/ structure:
# root_path/
# ├─ data/
# ├─ full/
# │  ├─ index
# | ...
#
# This should be the structure of the directory from extracting the
# downloaded tar file. For example, the folder extracted from
# trec05p-1.tgz should be trec05p-1/ which would have the desired structure
# of trec_root_path/


logger = logging.getLogger(__name__)

class TrecCorpusExtractor(BaseCorpusExtractor):

    expected_directory_structure = [
        {"name": "data", "type": "directory"},
        {"name": "full", "type": "directory", "children": [
            {"name": "index", "type": "file"}
        ]}
    ]

    def get_index(self):
        index_path = self.root_path / "full" / "index"
        logger.info(f"Fetching index file from {index_path}")
        index = pd.read_csv(index_path, sep=' ', dtype="string", header=None, 
                            names=["spam", "path"])
        index["spam"] = index["spam"].map({"spam": 1, "ham": 0})
        index["path"] = index["path"].str.removeprefix("../")
        return index
