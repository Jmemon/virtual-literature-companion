"""
a locally hosted web app that will go into the page_classifier_sidequest/page_dataset/ folder, and go through each json file, and display five pages of text at a time, with the label in bold below it. And a dropdown menu to select a different label if need be.
Then at the bottom of the page, there should be a button to accept the labels, and go on to the next five pages.

Any changes to the labels should be reflected in the json file.
"""
import json
import streamlit as st
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from virtual_literature_companion.processors.pdf2txt import PageType

dataset_dir = Path('page_classifier_sidequest/page_dataset')

@st.cache_data
def load_json_files():
    return sorted([f for f in dataset_dir.glob('*.json')])

def main():
    st.set_page_config(layout="wide")
    st.title('Page Dataset Labeler')
    json_files = load_json_files()
    if not json_files:
        st.warning('No JSON files found in page_dataset/')
        return
    selected_file = st.selectbox('Select book JSON', [f.name for f in json_files])
    if not selected_file:
        return
    json_path = dataset_dir / selected_file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'reset_counters' not in st.session_state:
        st.session_state.reset_counters = {}
    page_types = [pt.value for pt in PageType]
    batch_size = 3
    start = st.session_state.current_index
    end = min(start + batch_size, len(data))
    num_pages = end - start
    if num_pages == 0:
        return
    cols = st.columns(num_pages)
    for idx in range(num_pages):
        i = start + idx
        reset_key = f'{selected_file}_{i}'
        counter = st.session_state.reset_counters.get(reset_key, 0)
        page = data[i]
        with cols[idx]:
            st.subheader(f'Page {page["page"]}')
            num_lines = page['text'].count('\n') + 1
            height = max(100, num_lines * 30)  # 30 pixels per line for safety
            st.text_area('Text', page['text'], height=height, key=f'text_{i}', disabled=True)
            label_col1, label_col2 = st.columns([3,1])
            with label_col1:
                current_label = page.get('type', '')
                new_label = st.selectbox('Label', page_types, index=page_types.index(current_label) if current_label in page_types else 0, key=f'label_{i}_{counter}')
                if new_label != current_label:
                    data[i]['type'] = new_label
            with label_col2:
                if st.button('Set to Content', key=f'set_content_{i}_{counter}'):
                    data[i]['type'] = 'content'
                    st.session_state.reset_counters[reset_key] = counter + 1
                    st.rerun()
    st.markdown("""
    <style>
    div.stButton > button {
        font-size: 20px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button('Accept and Next', use_container_width=True):
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            st.session_state.current_index = end
            if end >= len(data):
                st.success(f'Finished labeling {selected_file}')
                st.session_state.current_index = 0
            st.rerun()

if __name__ == '__main__':
    main()