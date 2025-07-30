from pathlib import Path


REPO_DIR = Path(__file__).parent.parent

PAGE_CLASSIFIER_DIR = REPO_DIR / 'page_classifier'
PAGE_DATASET_DIR = PAGE_CLASSIFIER_DIR / 'page_dataset'
RAW_PAGE_DATASET_DIR = PAGE_DATASET_DIR / 'raw'

BOOK_SOURCE_DIR = REPO_DIR / 'book_source'