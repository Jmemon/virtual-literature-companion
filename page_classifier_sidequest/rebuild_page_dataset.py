import json
import os
import sys
from tqdm import tqdm

# Add the project root to the Python path to allow for package imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from virtual_literature_companion.llm.clean_text import clean_text

def process_file(filepath: str):
    """
    Reads a JSON file, cleans the 'text' field of each item synchronously,
    and writes the updated data back to the file.
    """
    print(f"Processing {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error reading or decoding {filepath}: {e}")
        return

    updated_data = []
    # Use tqdm for a progress bar, iterating over items
    for i, item in enumerate(tqdm(data, desc=f"Cleaning {os.path.basename(filepath)}")):
        cleaned_text = clean_text(item.get("text", ""))
        
        if cleaned_text is not None:
            item["text"] = cleaned_text
        else:
            print(f"    Warning: clean_text returned None for item {i} in {os.path.basename(filepath)}. Keeping original text.")
        updated_data.append(item)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=4)
    print(f"Finished processing {filepath}.")

def rebuild_page_dataset():
    """
    Finds all JSON files in the page_dataset directory and processes them.
    """
    page_dataset_dir = "page_classifier_sidequest/page_dataset"
    filepaths = [
        os.path.join(page_dataset_dir, filename)
        for filename in os.listdir(page_dataset_dir)
        if filename.endswith(".json")
    ]

    for filepath in filepaths:
        process_file(filepath)


if __name__ == "__main__":
    rebuild_page_dataset()
