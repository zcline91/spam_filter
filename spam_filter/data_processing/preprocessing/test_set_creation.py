import logging
from zlib import crc32

import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Create a test set using hashes. Idea courtesy of (and code adapted from)
# "Hands-On Machine Learning" by Aurélien Géron, O'Reilly Publishing
def test_set_check(identifier, test_ratio, string_id=False):
    if string_id:
        bytesobject = identifier.encode('utf-8')
    else:
        bytesobject = np.int64(identifier)
    return crc32(bytesobject) < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column, string_id=False, 
                           id_from_index=False):
    logger.info(f"Splitting dataframe into a training and test set by "
        f"'{id_column}' with test ratio {test_ratio}")
    if id_from_index:
        id_series = data.index.to_frame()[id_column]
    else:
        id_series = data[id_column]
    def test_set_checker(identifier):
        return test_set_check(identifier, test_ratio, string_id)
    in_test_set = id_series.apply(test_set_checker)
    return data.loc[~in_test_set], data.loc[in_test_set]
