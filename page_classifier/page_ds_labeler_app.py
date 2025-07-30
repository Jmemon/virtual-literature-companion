"""
Fast page labeling app for thousands of novel pages.
Features horizontal scrolling with number key shortcuts and auto-save functionality.
"""
import json
import streamlit as st
from pathlib import Path
import sys
import os
import time
from streamlit.components.v1 import html

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from virtual_literature_companion.processors.process_novel_text import PageType
from page_classifier.constants import RAW_PAGE_DATASET_DIR


def init_session_state():
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'label_count' not in st.session_state:
        st.session_state.label_count = 0
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'unverified_pages' not in st.session_state:
        st.session_state.unverified_pages = []
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None

def load_and_prepare_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add verified field if not present
    for page in data:
        if 'verified' not in page:
            page['verified'] = False
    
    # Save back with verified field
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return data

def get_unverified_pages(data):
    return [i for i, page in enumerate(data) if not page.get('verified', False)]

def save_tmp_file(data, filename):
    tmp_path = RAW_PAGE_DATASET_DIR / f"tmp_{filename}"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_final_file(filename):
    tmp_path = RAW_PAGE_DATASET_DIR / f"tmp_{filename}"
    final_path = RAW_PAGE_DATASET_DIR / filename
    if tmp_path.exists():
        with open(tmp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        tmp_path.unlink()  # Delete tmp file
        return True
    return False

def main():
    st.set_page_config(layout="wide", page_title="Page Dataset Labeler")
    st.title('📚 Fast Page Dataset Labeler')
    
    init_session_state()
    
    # Load available JSON files
    json_files = sorted([f for f in RAW_PAGE_DATASET_DIR.glob('*.json')])
    if not json_files:
        st.warning(f'No JSON files found in {RAW_PAGE_DATASET_DIR}')
        return
    
    # File selection
    selected_file = st.selectbox('Select book JSON', [f.name for f in json_files])
    if not selected_file:
        return
    
    # Load data when file changes
    if st.session_state.selected_file != selected_file:
        st.session_state.selected_file = selected_file
        json_path = RAW_PAGE_DATASET_DIR / selected_file
        st.session_state.data = load_and_prepare_data(json_path)
        st.session_state.unverified_pages = get_unverified_pages(st.session_state.data)
        st.session_state.current_index = 0
        st.session_state.label_count = 0
    
    data = st.session_state.data
    unverified_pages = st.session_state.unverified_pages
    
    if not unverified_pages:
        st.success("🎉 All pages have been verified!")
        return
    
    # Progress info
    total_pages = len(data)
    verified_count = total_pages - len(unverified_pages)
    st.info(f"Progress: {verified_count}/{total_pages} pages verified ({verified_count/total_pages*100:.1f}%)")
    
    page_types = [pt.value for pt in PageType]
    current_page_idx = unverified_pages[st.session_state.current_index]
    current_page = data[current_page_idx]
    
    # Create horizontal layout with center focus
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    # Left page (dimmed)
    with col_left:
        if st.session_state.current_index > 0:
            prev_idx = unverified_pages[st.session_state.current_index - 1]
            prev_page = data[prev_idx]
            st.markdown("### ← Previous")
            st.markdown('<div style="opacity: 0.4;">', unsafe_allow_html=True)
            st.text_area("Previous Page", prev_page['text'][:200] + "...", height=150, disabled=True, key="prev_text", label_visibility="hidden")
            st.markdown(f"**Current:** {prev_page.get('type', 'unknown')}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Center page (main focus)
    with col_center:
        st.markdown(f"### 📄 Page {current_page['page']} (Focus)")
        
        # Page text
        num_lines = current_page['text'].count('\n') + 1
        height = max(200, min(500, num_lines * 20))
        st.text_area("Page Content", current_page['text'], height=height, disabled=True, key="center_text")
        
        # Current label
        current_label = current_page.get('type', 'unknown')
        st.markdown(f"**Current Label:** `{current_label}`")
        
        # Number key instructions
        st.markdown("**Use number keys to label:**")
        for i, page_type in enumerate(page_types, 1):
            st.write(f"{i}. {page_type}")
    
    # Right page (dimmed)
    with col_right:
        if st.session_state.current_index < len(unverified_pages) - 1:
            next_idx = unverified_pages[st.session_state.current_index + 1]
            next_page = data[next_idx]
            st.markdown("### Next →")
            st.markdown('<div style="opacity: 0.4;">', unsafe_allow_html=True)
            st.text_area("Next Page", next_page['text'][:200] + "...", height=150, disabled=True, key="next_text", label_visibility="hidden")
            st.markdown(f"**Current:** {next_page.get('type', 'unknown')}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Hidden buttons for keyboard shortcuts
    st.markdown('<div style="display: none;">', unsafe_allow_html=True)
    cols = st.columns(len(page_types))
    for i, page_type in enumerate(page_types):
        with cols[i]:
            if st.button(f"Label as {page_type}", key=f"hidden_btn_{page_type}"):
                data[current_page_idx]['type'] = page_type
                data[current_page_idx]['verified'] = True
                st.session_state.label_count += 1

                if st.session_state.label_count % 10 == 0:
                    save_tmp_file(data, selected_file)
                
                st.session_state.unverified_pages = get_unverified_pages(data)
                if st.session_state.current_index >= len(st.session_state.unverified_pages):
                    st.session_state.current_index = max(0, len(st.session_state.unverified_pages) - 1)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # JavaScript for number key handling
    js_code = f"""
    <script>
    document.addEventListener('keydown', function(event) {{
        const key = event.key;
        const pageTypes = {json.dumps(page_types)};
        
        if (key >= '1' && key <= '9') {{
            const index = parseInt(key) - 1;
            if (index < pageTypes.length) {{
                const pageType = pageTypes[index];
                const button = window.parent.document.querySelector(`button[data-testid='stButton'][kind='secondary'][key='hidden_btn_${{pageType}}']`);
                if (button) {{
                    button.click();
                }}
            }}
        }}
    }});
    </script>
    """
    html(js_code, height=0)
    
    # Control buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⏮️ Previous", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.rerun()
    
    with col2:
        # Manual label selection
        new_label = st.selectbox("Manual Label", page_types, 
                                index=page_types.index(current_label) if current_label in page_types else 0,
                                key="manual_label")
        
        if st.button("✅ Apply Label"):
            data[current_page_idx]['type'] = new_label
            data[current_page_idx]['verified'] = True
            st.session_state.label_count += 1
            
            # Auto-save every 10 labels
            if st.session_state.label_count % 10 == 0:
                save_tmp_file(data, selected_file)
                st.success(f"Auto-saved after {st.session_state.label_count} labels!")
            
            # Move to next unverified page
            st.session_state.unverified_pages = get_unverified_pages(data)
            if st.session_state.current_index >= len(st.session_state.unverified_pages):
                st.session_state.current_index = max(0, len(st.session_state.unverified_pages) - 1)
            
            st.rerun()
    
    with col3:
        if st.button("⏭️ Next", disabled=st.session_state.current_index >= len(unverified_pages) - 1):
            st.session_state.current_index += 1
            st.rerun()
    
    with col4:
        if st.button("💾 Save to Final"):
            save_tmp_file(data, selected_file)  # Save current state to tmp first
            if save_final_file(selected_file):
                st.success("✅ Saved to final JSON file!")
            else:
                st.error("❌ No tmp file to save")
    
    # Statistics
    st.markdown("---")
    st.markdown(f"**Session Stats:** {st.session_state.label_count} labels applied | {len(unverified_pages)} pages remaining")

if __name__ == '__main__':
    main()