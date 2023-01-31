#!/usr/bin/env python3 

import sys
import tarfile
import io
from contextlib import redirect_stdout
from pathlib import Path


import pandas as pd


"""This is a helper script for creating an index csv file of the 
email files contained in the Ling corpus, and whether or not each
is spam or ham. Call this script with a single argument: the 
directory of the ling tarball."""


SCRIPTLOCATION = Path.cwd().joinpath(Path(__file__).parent).resolve()
ROOTLOCATION = SCRIPTLOCATION.parent # Set parent directory as root
INDEXLOCATION = ROOTLOCATION / 'data_processing' / 'corpus' / '_ling_index.csv'


path = sys.argv[1]
f = io.StringIO()
with tarfile.open(path) as t, redirect_stdout(f):
    t.list()
f.seek(0)
df = pd.read_csv(f, header=None, delim_whitespace=True)
files = (df[5]
    [~df[5].str.endswith('/')] # exclude directories
    .str.removeprefix('lingspam_public/') # remove root prefix
    .where(lambda x: x.str.startswith('bare/')) # set files outside of bare to na
    .dropna() # and get rid of them
)
spam = (files
    .str.rsplit('/', n=1) # where filename
    .str[-1].str.startswith('spmsg') # starts with 'spmsg'
    .astype('int8')
)
spam_and_files = pd.DataFrame({"spam": spam, "path": files})
spam_and_files.to_csv(INDEXLOCATION, index=False)
