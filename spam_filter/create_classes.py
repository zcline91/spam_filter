import logging

from data_processing import CORPORA_CSV_PATH, CORPUS_FILENAMES, \
    TEST_RATIO, SPAM_CLASS_PATH, SPAM_CLASS_FILENAMES, load_train_test_csvs


handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(name)s [%(levelname)s] - %(message)s")
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)


def save_train_test_classes(csv_path=CORPORA_CSV_PATH, 
        corpus_names=CORPUS_FILENAMES, test_ratio=TEST_RATIO,
        output_path=SPAM_CLASS_PATH, output_files=SPAM_CLASS_FILENAMES):
    train_set, test_set = load_train_test_csvs(csv_path, corpus_names, 
        test_ratio)
    train_set['spam'].to_csv(output_path / output_files['train'])
    test_set['spam'].to_csv(output_path / output_files['test'])


if __name__ == '__main__':
    save_train_test_classes()
