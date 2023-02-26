#!/usr/bin/env python3

import logging
import argparse
from pathlib import Path

from data_processing.preprocessing import load_train_test_csvs, create_docbins
from data_processing import CORPORA_CSV_PATH, CORPUS_FILENAMES, DOCBIN_PATH, \
    DOCBIN_FILENAMES


# Set up logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def existing_directory(string):
    """A helper function for the arguments parser.
    Checks if the string specifies an existing directory from the 
    current path."""
    path = Path(string)
    if not (path.exists() and path.is_dir()):
        raise NotADirectoryError(string)
    return path


def get_arguments():
    """A function for collecting command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=existing_directory,
                        default=CORPORA_CSV_PATH,
                        help="directory containing the corpus csv files")
    parser.add_argument('--set', choices=["train", "test", "all"], 
                        default="all",
                        help="the email set to run the spacy pipeline on")
    parser.add_argument('--field', choices=["body", "subject", "all"], 
                        default="all",
                        help="the field to run the spacy pipeline on")
    parser.add_argument('--batchsize', type=int, default=2000)
    parser.add_argument('-d', '--output_dir', type=existing_directory, 
                        default=DOCBIN_PATH,
                        help="the directory to output the docbin file(s)")
    parser.add_argument('-F', '--force', action='store_true',
                        help="force output file(s) to be overwritten")
    verbosegroup = parser.add_mutually_exclusive_group()
    verbosegroup.add_argument('-v', '--verbose', action='store_true',
                              help=("verbose mode - show extra log info "
                                "(debug level)"))
    verbosegroup.add_argument('-q', '--quiet', action='store_true',
                              help=("quiet mode - show minimal log info "
                                "(error level)"))
    parser.add_argument('-l', '--log',
                        help=("a filename to store the log rather than "
                            "outputting to the console"))
    return parser.parse_args()


def parse_arguments(args):
    """Parse command-line arguments and return 
    - the path of the corpus csv files
    - a list of tuples of the form (data_set [train/test], 
      field [body/subject], path) where the spacy docbin created for each 
      given data_set and field will be saved in path.
    - the batch size for the docbins"""
    # Check if output directory exists
    if not args.output_dir.exists():
        raise FileNotFoundError(f"{args.output_dir} does not exist")
    # Check if corpora are all present
    for corpus, filename in CORPUS_FILENAMES.items():
        corpus_path = (args.corpus_dir / filename)
        if not corpus_path.exists():
            raise FileNotFoundError(f"{corpus} file {corpus_path} not found")
    # Create list of data sets, fields, and paths
    data_sets = ["train", "test"] if args.set == "all" else [args.set]
    fields = ["body", "subject"] if args.field == "all" else [args.field]
    set_field_path_list = [
        (data_set, field, 
            (args.output_dir / DOCBIN_FILENAMES[data_set][field])) 
        for data_set in data_sets 
        for field in fields
    ]
    # Only overwrite an existing file if the force filename flag (-F) is used.
    if not args.force:
        for _, _, path in set_field_path_list:
            if path.exists() or path.with_suffix('.spacy').exists():
                raise FileExistsError(f"{path} already exists. Use "
                    "option '-F' if you would like to overwrite this file.")
    # Set which handler to use for logging
    if args.log:
        handler = logging.FileHandler(args.log)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s [%(levelname)s] - %(message)s")
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(name)s [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    # Use verbosity level to set logging level
    if args.verbose:
        loglevel = logging.DEBUG
    elif args.quiet:
        loglevel = logging.ERROR
    else:
        loglevel = logging.INFO
    handler.setLevel(loglevel)
    # Add the handler to the root logger
    logging.getLogger().addHandler(handler)
    return args.corpus_dir, set_field_path_list, args.batchsize


def main():
    args = get_arguments()
    corpus_dir, set_field_path_list, batch_size = parse_arguments(args)
    train_set, test_set = load_train_test_csvs(corpus_dir, 
        corpus_names=CORPUS_FILENAMES)
    data_sets = {'train': train_set, 'test': test_set}
    for data_set_name, field, path in set_field_path_list:
        logger.info(f"Creating docbin for {data_set_name}[{field}] and "
            f"storing in {path}")
        create_docbins(data_sets[data_set_name][field], path, 
            batch_size=batch_size)


if __name__ == '__main__':
    main()
    