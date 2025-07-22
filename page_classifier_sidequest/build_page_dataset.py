"""
Go through the books in book_pdfs, and for each book, extract the text from each page, classify each page using the method in pdf2txt.py, and save the results to a json file as a list of dictionaries, where each dictionary contains the page text, and the page type.

The jsons should go in page_classifier_sidequest/page_dataset/
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

from virtual_literature_companion.processors.page_extraction import extract_single_page_text
from virtual_literature_companion.processors.process_novel_text import categorize_page
from virtual_literature_companion.llm.clean_text import clean_text_async


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
            futures = {executor.submit(extract_single_page_text, str(pdf_path), i): i for i in range(total_pages)}
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

async def update_datasets_clean_text(
        book_pdfs_dir: str = 'book_pdfs', output_dir: str = 'page_classifier_sidequest/page_dataset'
):
    """
    Rebuild datasets but clean the text using the clean_text function. Leave the existing page types.
    We are assuming that the pages we re-extract from the pdfs will line up one-to-one with the existing json dicts.
    """
    book_pdfs_dir = Path(book_pdfs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not book_pdfs_dir.exists():
        print(f'Warning: Directory {book_pdfs_dir} does not exist. Please add PDF files there.')
        return

    for pdf_path in book_pdfs_dir.glob('*.pdf'):
        novel_name = pdf_path.stem
        json_path = output_dir / f'{novel_name}.json'
        if not json_path.exists():
            print(f'Warning: No dataset found for {novel_name}. Please run build_dataset() first.')
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_dataset = json.load(f)
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        page_texts = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(extract_single_page_text, str(pdf_path), i): i for i in range(total_pages)}
            for future in as_completed(futures):
                page_num, text = future.result()
                page_texts[page_num] = text

        dataset = []
        sorted_pages_nums = sorted(page_texts.keys())
        batch_size = 20
        pbar = tqdm(total=len(sorted_pages_nums), desc=f'Cleaning {novel_name}')

        for i in range(0, len(sorted_pages_nums), batch_size):
            batch_page_nums = sorted_pages_nums[i:i+batch_size]
            
            tasks = [clean_text_async(page_texts[pn], max_tokens=30_000) for pn in batch_page_nums]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            cleaned_texts_batch = []
            for page_num, result in zip(batch_page_nums, results):
                if isinstance(result, Exception) or result is None:
                    cleaned_texts_batch.append(f"{page_texts[page_num]}")
                    print(f"ERROR: {result}")
                else:
                    cleaned_texts_batch.append(result)

            pre_update_len = len(dataset)
            for page_num, cleaned_text in zip(batch_page_nums, cleaned_texts_batch):
                page_type = existing_dataset[page_num]['type']
                dataset.append({
                    'page': page_num + 1,
                    'raw_text': page_texts[page_num],
                    'text': cleaned_text,
                    'type': page_type
                })
            pbar.update(len(dataset) - pre_update_len)
        pbar.close()

        # Find the highest existing version number and increment it
        base_name = f'{novel_name}'
        existing_versions = []
        for existing_file in output_dir.glob(f'{base_name}*.json'):
            stem = existing_file.stem
            if stem == base_name:
                existing_versions.append(0)  # Original file has version 0
            elif stem.startswith(f'{base_name}_') and stem[len(f'{base_name}_'):].isdigit():
                version_num = int(stem[len(f'{base_name}_'):])
                existing_versions.append(version_num)
        
        next_version = max(existing_versions, default=-1) + 1
        if next_version == 0:
            json_filename = f'{novel_name}.json'
        else:
            json_filename = f'{novel_name}_{next_version}.json'
        
        json_path = output_dir / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        print(f'Saved dataset for {novel_name} to {json_path}')

if __name__ == '__main__':
    asyncio.run(update_datasets_clean_text())