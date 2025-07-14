"""
Go through the books in book_pdfs, and for each book, extract the text from each page, classify each page using the method in pdf2txt.py, and save the results to a json file as a list of dictionaries, where each dictionary contains the page text, and the page type.

The jsons should go in page_classifier_sidequest/page_dataset/
"""
import json
import os
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdfplumber
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from virtual_literature_companion.processors.pdf2txt import extract_page_text, categorize_page, PageType, calculate_word_stats

def build_dataset(book_pdfs_dir: str = 'book_pdfs', output_dir: str = 'page_classifier_sidequest/page_dataset'):
    book_pdfs_dir = Path(book_pdfs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not book_pdfs_dir.exists():
        print(f'Warning: Directory {book_pdfs_dir} does not exist. Please add PDF files there.')
        return
    for pdf_path in book_pdfs_dir.glob('*.pdf'):
        novel_name = pdf_path.stem
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        page_texts = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_page_text, str(pdf_path), i): i for i in range(total_pages)}
            for future in as_completed(futures):
                page_num, text = future.result()
                page_texts[page_num] = text
        word_counts = [calculate_word_stats(text)[0] for text in page_texts.values() if text.strip()]
        mean_words = statistics.mean(word_counts) if word_counts else 0
        std_words = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        dataset = []
        for page_num in sorted(page_texts.keys()):
            text = page_texts[page_num]
            page_type = categorize_page(text, page_num, total_pages, mean_words, std_words)
            dataset.append({
                'page': page_num + 1,
                'text': text,
                'type': page_type.value
            })
        json_path = output_dir / f'{novel_name}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        print(f'Saved dataset for {novel_name} to {json_path}')
if __name__ == '__main__':
    build_dataset()