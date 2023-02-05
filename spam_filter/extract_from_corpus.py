#!/usr/bin/env python3

import logging
import argparse
from pathlib import Path


from data_processing.corpus import (EnronCorpusExtractor, LingCorpusExtractor, 
    TrecCorpusExtractor)
from data_processing import CORPORA_CSV_PATH, CORPUS_FILENAMES


EXTRACTORS = {'enron': EnronCorpusExtractor, 'ling': LingCorpusExtractor,
    'trec': TrecCorpusExtractor}


class ArgumentError(Exception):
    pass


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-t', '--type',
                        choices=['enron', 'ling', 'trec'],
                        help="the type of corpus to extract")
    mode.add_argument('--all', action='store_true', 
                        help=("extract from all corpora subdirectories in the "
                            "passed data_root_path"))
    parser.add_argument('data_root_path', type=existing_directory,
                        help="the root directory of the corpus (or corpora)")
    parser.add_argument('-d', '--output_dir', type=existing_directory, 
                        default=CORPORA_CSV_PATH,
                        help="the directory to output csv file")
    parser.add_argument('-f', '--filename', metavar="OUTPUT_FILE",
                        help=("filename for output csv file (only if "
                            "`type` is not `all`)"))
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
    """Parse command-line arguments and return a list of tuples 
    of the form (BaseCorpusExtractor, output_path) where each 
    corpus extractor has been instatiated with a root_path based 
    on the arguments provided."""
    extractor_path_list = []
    if args.all:
        if args.filename:
            raise ArgumentError("In 'all' mode, the "
                "filename cannot be manually set.")
        # Check if each subdirectory is a corpus of any of the known types, 
        # by comparing the start of the directory name with 'enron', 'ling', 
        # and 'trec'. If so, initiate the appropriate extractor, and add it 
        # to extractor_path_list along with an appropriate output_path.
        for child in args.data_root_path.iterdir():
            if child.is_dir():
                for corpus_type, extractor_class in EXTRACTORS.items():
                    if child.name.startswith(corpus_type):
                        extractor = extractor_class(child) 
                        break
                else:
                    continue
                corpus_filename = next(filename 
                    for name, filename in CORPUS_FILENAMES.items()
                    if child.name.startswith(name))
                output_path = args.output_dir / corpus_filename
                extractor_path_list.append((extractor, output_path))
    else:
        extractor_class = EXTRACTORS[args.type]
        extractor = extractor_class(args.data_root_path)
        corpus_filename = next(filename 
            for name, filename in CORPUS_FILENAMES.items() 
            if args.data_root_path.name.startswith(name))
        filename = args.filename or corpus_filename
        output_path = args.output_dir / filename
        extractor_path_list.append((extractor, output_path))
    # Only overwrite an existing file if the force filename flag (-F) is used.
    if not args.force:
        for _, output_path in extractor_path_list:
            if output_path.exists():
                raise FileExistsError(f"{output_path} already exists. Use "
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
    return extractor_path_list


def main():
    args = get_arguments()
    extractor_path_list = parse_arguments(args)
    for extractor, output_path in extractor_path_list:
        extractor.create_csv(output_path)


if __name__ == "__main__":
    main()
