"""
Go through the books in book_pdfs, and for each book, extract the text from each page, classify each page using the method in pdf2txt.py, and save the results to a json file as a list of dictionaries, where each dictionary contains the page text, and the page type.

The jsons should go in page_classifier/page_dataset/
"""
import json
import os
import re
from typing import Tuple
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdfplumber
from tqdm import tqdm
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from virtual_literature_companion.processors.pdf_extraction import extract_single_page_text
from virtual_literature_companion.processors.epub_extraction import extract_page_text as epub_extract_page_text
from virtual_literature_companion.processors.process_novel_text import categorize_page
from virtual_literature_companion.llm.clean_text import clean_text_async

from page_classifier.constants import RAW_PAGE_DATASET_DIR

def _extract_text_from_book(book_path: Path) -> Tuple[dict, int]:
    """Helper to extract text from PDF or EPUB."""
    page_texts = {}
    total_pages = 0
    if book_path.suffix == '.pdf':
        with pdfplumber.open(book_path) as pdf:
            total_pages = len(pdf.pages)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_single_page_text, str(book_path), i): i for i in range(total_pages)}
            for future in as_completed(futures):
                page_num, text = future.result()
                page_texts[page_num] = text
    elif book_path.suffix == '.epub':
        page_texts, total_pages = epub_extract_page_text(str(book_path))

    return page_texts, total_pages


def calculate_word_stats(text: str) -> Tuple[int, float]:
    """
    Calculate word count and word diversity for text analysis.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        Tuple[int, float]: (word_count, word_diversity_ratio)
            word_diversity_ratio is unique_words / total_words
    """
    # Clean and split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    if word_count == 0:
        return 0, 0.0
    
    # Calculate diversity as ratio of unique words to total words
    unique_words = len(set(words))
    diversity_ratio = unique_words / word_count
    
    return word_count, diversity_ratio

async def build_dataset(book_source_dir: str = 'book_source', output_dir: str = str(RAW_PAGE_DATASET_DIR)):
    book_source_dir = Path(book_source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not book_source_dir.exists():
        print(f'Warning: Directory {book_source_dir} does not exist. Please add book files there.')
        return

    book_paths = list(book_source_dir.glob('*.pdf')) + list(book_source_dir.glob('*.epub'))
    for book_path in book_paths:
        novel_name = book_path.stem
        json_path = output_dir / f'{novel_name}.json'
        if json_path.exists():
            print(f"Skipping {novel_name} as it's already been parsed.")
            continue
        
        page_texts, total_pages = _extract_text_from_book(book_path)

        # Clean texts
        cleaned_texts = {}
        sorted_pages_nums = sorted(page_texts.keys())
        batch_size = 20
        pbar = tqdm(total=len(sorted_pages_nums), desc=f'Cleaning {novel_name}')

        for i in range(0, len(sorted_pages_nums), batch_size):
            batch_page_nums = sorted_pages_nums[i:i+batch_size]
            
            tasks = [clean_text_async(page_texts[pn], max_tokens=20_000) for pn in batch_page_nums]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for page_num, result in zip(batch_page_nums, results):
                if isinstance(result, Exception) or result is None:
                    cleaned_texts[page_num] = page_texts[page_num]
                    print(f"ERROR cleaning page {page_num} for {novel_name}: {result}")
                else:
                    cleaned_texts[page_num] = result
            pbar.update(len(batch_page_nums))
        pbar.close()

        word_counts = [calculate_word_stats(text)[0] for text in page_texts.values() if text.strip()]
        mean_words = statistics.mean(word_counts) if word_counts else 0
        std_words = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        dataset = []
        for page_num in sorted(page_texts.keys()):
            raw_text = page_texts[page_num]
            cleaned_text = cleaned_texts.get(page_num, raw_text)
            page_type = categorize_page(raw_text, page_num, total_pages, mean_words, std_words)
            dataset.append({
                'page': page_num + 1,
                'raw_text': raw_text,
                'text': cleaned_text,
                'type': page_type.value
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        print(f'Saved dataset for {novel_name} to {json_path}')


if __name__ == '__main__':
    asyncio.run(build_dataset())
