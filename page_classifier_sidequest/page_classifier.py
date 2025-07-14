"""
Using the labeled page dataset, build a simple page classifier that will effectively classify the pages of a novel.
This will use a combination of global statistics about the pages, and the text on the page.

We will build the classifier by running an llm in a loop, where the llm will be asked to create a classification function, we will run it on a subset of the dataset, and if it does not perform well, we will store which pages it failed on, show the llm and have it improve its classifier.
"""

import json
import random
import importlib.util
import os
import sys
from pathlib import Path
import statistics
import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from virtual_literature_companion.ai import make_llm_request
from virtual_literature_companion.processors.pdf2txt import PageType

dataset_dir = Path('page_classifier_sidequest/page_dataset')
classifier_file = Path('page_classifier_sidequest/generated_classifier.py')

classifier_fn_signature = "def classify_page(text: str, page_num: int, total_pages: int, mean_words: float, std_words: float) -> str" 

def load_labeled_data():
    all_pages = []
    for json_path in dataset_dir.glob('*.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        for page in data:
            if 'type' in page and page['type']:  # Assume labeled if type is set
                all_pages.append(page)
    return all_pages

def generate_classifier_code(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    response = make_llm_request(messages, max_tokens=2000, temperature=0.7)
    return response.strip()

def test_classifier(code, pages, mean_words, std_words):
    # Write code to file
    with open(classifier_file, 'w') as f:
        f.write(code)
    try:
        # Import
        spec = importlib.util.spec_from_file_location('generated_classifier', classifier_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classify = module.classify_page
    except Exception as e:
        return 0, [], {}, str(e)
    actual_counts = Counter(p['type'] for p in pages)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    failures = []
    for page in pages:
        actual = page['type']
        try:
            pred = classify(page['text'], page['page'] - 1, 100, mean_words, std_words)  # Assume total_pages=100 for sim
            if pred == actual:
                tp[actual] += 1
            else:
                fp[pred] += 1
                fn[actual] += 1
            failures.append({'page': page['page'], 'text': page['text'][:200], 'predicted': pred, 'actual': actual})
        except Exception as e:
            fn[actual] += 1
            failures.append({'page': page['page'], 'text': page['text'][:200], 'error': str(e), 'actual': actual})
    correct = sum(tp.values())
    accuracy = correct / len(pages) if pages else 0
    # Compute metrics
    all_classes = [pt.value for pt in PageType]
    metrics = {}
    for c in all_classes:
        count = actual_counts[c]
        if count == 0:
            metrics[c] = {'precision': None, 'recall': None, 'f1': None, 'count': 0}
        else:
            denom_p = tp[c] + fp[c]
            denom_r = tp[c] + fn[c]
            prec = tp[c] / denom_p if denom_p > 0 else 0.0
            recall = tp[c] / denom_r if denom_r > 0 else 0.0
            f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
            metrics[c] = {'precision': prec, 'recall': recall, 'f1': f1, 'count': count}
    return accuracy, failures, metrics, None

def main():
    data = load_labeled_data()
    if not data:
        print('No labeled data found.')
        return
    # Compute dummy global stats
    word_counts = [len(p['text'].split()) for p in data]
    mean_words = statistics.mean(word_counts)
    std_words = statistics.stdev(word_counts)
    # Split data
    random.shuffle(data)
    test_size = int(len(data) * 0.2)
    test_pages = data[:test_size]
    train_pages = data[test_size:]

    # Create a timestamped directory to save iterations of generated classifiers
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gen_dir = Path('page_classifier_sidequest/generated_classifiers') / timestamp
    gen_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1
    best_code = None
    best_metrics = None

    # Initial prompt
    # Group pages by type to ensure balanced examples
    pages_by_type = {}
    for page in train_pages:
        page_type = page['type']
        if page_type not in pages_by_type:
            pages_by_type[page_type] = []
        pages_by_type[page_type].append(page)
    
    # Select up to 2 examples from each page type, preferably from different books
    selected_examples = []
    for page_type, pages in pages_by_type.items():
        # Group by book to try to get examples from different books
        pages_by_book = {}
        for page in pages:
            book_name = page.get('book', 'unknown')
            if book_name not in pages_by_book:
                pages_by_book[book_name] = []
            pages_by_book[book_name].append(page)
        
        # Select up to 2 examples, preferring different books
        type_examples = []
        books_used = set()
        
        # First pass: try to get one example from each book
        for book_name, book_pages in pages_by_book.items():
            if len(type_examples) >= 2:
                break
            if book_name not in books_used:
                type_examples.append(random.choice(book_pages))
                books_used.add(book_name)
        
        # Second pass: fill remaining slots if needed
        if len(type_examples) < 2:
            remaining_pages = [p for p in pages if p not in type_examples]
            additional_needed = min(2 - len(type_examples), len(remaining_pages))
            if additional_needed > 0:
                type_examples.extend(random.sample(remaining_pages, additional_needed))
        
        selected_examples.extend(type_examples)
    
    examples = '\n'.join([f"Text: {p['text'][:200]}... Label: {p['type']}" for p in selected_examples])
    prompt = f"""Write a Python function with this signature: {classifier_fn_signature}

It should classify book pages into one of: {', '.join([pt.value for pt in PageType])}. Use heuristics, regex, and stats.

Examples:
{examples}

Output the python code in <python>...</python> tags. Make sure you include all necessary imports, check for syntax errors, and any other problems. This code should be ready to run. Describe the logic of your solution and why you think it will work in a few sentences, placed in <description>...</description> tags."""
    output = generate_classifier_code(prompt)
    code = output.split('<python>')[1].split('</python>')[0].strip()
    description = output.split('<description>')[1].split('</description>')[0].strip()

    iter_file = gen_dir / f"generated_classifier_{0}.py"
    with open(iter_file, 'w') as f:
        f.write(code)

    iteration = 0
    max_iterations = 10
    target_accuracy = 0.95
    while iteration < max_iterations:
        accuracy, failures, metrics, error = test_classifier(code, test_pages, mean_words, std_words)
        if error is not None:
            print(f'Error in classifier: {error}')
            prompt = f"""The classifier code had an error: {error}.

Current code:
{code}

Description of the current code:
{description}

Fix the code to resolve the error.

Output the fixed python code in <python>...</python> tags. Make sure you include all necessary imports, check for syntax errors, and any other problems. This code should be ready to run. Describe the logic of your solution and why you think it will work in a few sentences, placed in <description>...</description> tags."""
            output = generate_classifier_code(prompt)
            code = output.split('<python>')[1].split('</python>')[0].strip()
            description = output.split('<description>')[1].split('</description>')[0].strip()
            iter_file = gen_dir / f"generated_classifier_{iteration}_fixed.py"
            with open(iter_file, 'w') as f:
                f.write(code)
            continue
        # Compute score
        f1_toc = metrics.get('table_of_contents', {}).get('f1') or 0
        f1_chap = metrics.get('chapter_start', {}).get('f1') or 0
        f1_cont = metrics.get('content', {}).get('f1') or 0
        score = f1_toc + f1_chap + f1_cont
        print(f'Iteration {iteration}: Accuracy {accuracy:.4f}')
        print('Per-class metrics:')
        for c, m in metrics.items():
            if m['count'] == 0:
                print(f'  {c}: No examples')
            else:
                print(f'  {c}: count={m["count"]}, precision={m["precision"]*100 if m["precision"] is not None else "N/A"}%, recall={m["recall"]*100 if m["recall"] is not None else "N/A"}%, f1={m["f1"]*100 if m["f1"] is not None else "N/A"}%')
        if score > best_score:
            best_score = score
            best_code = code
            best_metrics = metrics
        if accuracy >= target_accuracy:
            print('Target accuracy reached.')
            break
        failure_examples = '\n'.join([f"Text: {f['text']} " + (f"Predicted: {f['predicted']} Actual: {f['actual']}" if 'predicted' in f else f"Error: {f['error']} Actual: {f['actual']}") for f in failures[:5]])
        prompt = f"""Improve this classifier function. It failed on these: {failure_examples}.

Current code:
{code}

Description of the current code:
{description}

Output the improved python code in <python>...</python> tags. Make sure you include all necessary imports, check for syntax errors, and any other problems. This code should be ready to run. Describe the logic of your solution and why you think it will work in a few sentences, placed in <description>...</description> tags."""
        output = generate_classifier_code(prompt)
        code = output.split('<python>')[1].split('</python>')[0].strip()
        description = output.split('<description>')[1].split('</description>')[0].strip()
        iter_file = gen_dir / f"generated_classifier_{iteration + 1}.py"
        with open(iter_file, 'w') as f:
            f.write(code)
        iteration += 1
    with open('page_classifier_sidequest/final_classifier.py', 'w') as f:
        f.write(best_code)
    print('Final classifier saved. Best score:', best_score)

if __name__ == '__main__':
    main()
