from enum import Enum
from pathlib import Path
import json

from virtual_literature_companion.processors.process_novel_text import PageType


class NewPageTypes(Enum):
    BLANK = "blank"
    TITLE_PAGE = "title_page"
    TABLE_OF_CONTENTS = "table_of_contents"
    COPYWRIGHT_PAGE = "copyright_page"
    STORY_BREAK = "story_break"
    FRONT_MATTER_BREAK = "front_matter_break"
    BACK_MATTER_BREAK = "back_matter_break"
    CONTENT = "content"

page_types_map = {
    PageType.TITLE_PAGE: NewPageTypes.TITLE_PAGE,
    PageType.COPYRIGHT_PAGE: NewPageTypes.COPYWRIGHT_PAGE,
    PageType.DEDICATION_PAGE: NewPageTypes.FRONT_MATTER_BREAK,
    PageType.TABLE_OF_CONTENTS_PAGE: NewPageTypes.TABLE_OF_CONTENTS,
    PageType.FOREWORD_PREFACE_START: NewPageTypes.FRONT_MATTER_BREAK,
    PageType.ACKNOWLEDGEMENTS_START: NewPageTypes.FRONT_MATTER_BREAK,
    PageType.INTRODUCTION_START: NewPageTypes.FRONT_MATTER_BREAK,
    PageType.CHAPTER_START: NewPageTypes.STORY_BREAK,
    PageType.PART_START: NewPageTypes.STORY_BREAK,
    PageType.CONTENT: NewPageTypes.CONTENT,
    PageType.APPENDIX_START: NewPageTypes.BACK_MATTER_BREAK,
    PageType.GLOSSARY_START: NewPageTypes.BACK_MATTER_BREAK,
    PageType.BIBLIOGRAPHY_PAGE: NewPageTypes.BACK_MATTER_BREAK,
    PageType.INDEX_PAGE: NewPageTypes.BACK_MATTER_BREAK,
    PageType.THROWAWAY: NewPageTypes.BLANK,
}

def simplify_page_categories(output_dir: str = 'page_classifier_sidequest/page_dataset') -> None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f'Warning: Directory {output_dir} does not exist.')
        return

    for json_path in output_dir.glob('*.json'):
        novel_name = json_path.stem
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_dataset = json.load(f)

        updated_dataset = []
        for page_data in existing_dataset:
            old_page_type = PageType(page_data['type'])
            new_page_type = page_types_map.get(old_page_type, NewPageTypes.CONTENT) 
            
            page_data['type'] = new_page_type.value
            updated_dataset.append(page_data)

        base_name = f'{novel_name}'
        existing_versions = []
        for existing_file in output_dir.glob(f'{base_name}*.json'):
            stem = existing_file.stem
            if stem == base_name:
                existing_versions.append(0)
            elif stem.startswith(f'{base_name}_') and stem[len(f'{base_name}_'):].isdigit():
                version_num = int(stem[len(f'{base_name}_'):])
                existing_versions.append(version_num)
        
        next_version = max(existing_versions, default=-1) + 1
        if next_version == 0:
            json_filename = f'{novel_name}.json'
        else:
            json_filename = f'{novel_name}_{next_version}.json'
        
        new_json_path = output_dir / json_filename
        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_dataset, f, indent=4, ensure_ascii=False)
        print(f'Saved simplified dataset for {novel_name} to {new_json_path}')

if __name__ == '__main__':
    simplify_page_categories()

