#!/usr/bin/env python3 

import sys
import tarfile
import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


"""This is a helper script for creating an index csv file of the 
email files contained in the Enron corpus, and whether or not each
is spam or ham. Call this script with a single argument: a directory
containing the raw tarballs from the Enron corpus."""


SCRIPTLOCATION = Path.cwd().joinpath(Path(__file__).parent).resolve()
ROOTLOCATION = SCRIPTLOCATION.parent # Set parent directory as root
INDEXLOCATION = ROOTLOCATION / 'data_processing' / 'corpus' / '_enron_index.csv'


root_path = Path(sys.argv[1])
spam_and_files = pd.DataFrame({"spam": [], "path": []})
for path in root_path.iterdir():
    f = io.StringIO()
    with tarfile.open(path) as t, redirect_stdout(f):
        t.list()
    f.seek(0)
    df = pd.read_csv(f, header=None, delim_whitespace=True, dtype=str)
    rawfiles = (df[5]
        [~df[5].str.endswith('/')] # remove directories from list
    )
    files = 'raw/' + rawfiles
    spam = (rawfiles
        .str.split('/', n=1) # where containing directory
        .str[0].isin(('BG', 'GP', 'SH')) # is in 'BG', 'GP', 'SH
    )
    partial_spam_and_files = pd.DataFrame({"spam": spam, "path": files})
    spam_and_files = pd.concat([spam_and_files, partial_spam_and_files],
                               ignore_index=True)
spam_and_files['spam'] = spam_and_files['spam'].astype('int8')
spam_and_files.to_csv(INDEXLOCATION, index=False)
