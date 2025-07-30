


==== PDF ingestion is different than EPUB ingestion. ====
We're going to abandon pdf extraction for now. Since:
 - most e-reading devices are epub 
 - all the public domain books are available as epubs
 - AND its an easier format to ingest

PDF ingestion:
 - pdf2json: pdf file -> json (list of dicts) where each dict is a page from the book.
 - clean_text: json -> json (list of dicts) where each dict is a page from the book. Page headings removed, OCR errors (if that kind of pdf), etc. The pages will be a lot less text than the epubs, so can do more transofrmation more reliably.
 - structure_novel: json -> json (list of dicts) where each dict is a section of the book, with each section classified (chapter, dedication, acknowledgements, etc.)
 - restructure_json: json -> json; where we have keys for each section of the book, and the values are the existing jsons for each section. So list(keys) should be something like a table of contents, except it will include front matter, back matter, title page, etc.

EPUB ingestion:
Output format: json with keys for each section of the book (title_page, table_of_contents, dedication, chapter X, etc.) mapped dicts for the section's content (raw and cleaned) as well as section type
 - epub2json: epub file -> json (section_type to dict) where each dict is a section of the book (toc, dedication, chapter, etc.). Ie we don't need to do furhter processing to extract this structure, since its baked into the epub format. We are extracting from the file and classifying sections here. Raw text goes under `raw_text` field.
 - clean_text: json -> json The text will already be pretty clean, but for this we should prioritize getting rid of gutenberg headings + it will add in some nice formatting in some places. Add a `clean_text` field to each section dict for the cleaned text.

==== At this point both file formats have been unified to the same json format. ====

From here we chunk (keeping track of chunk positions), embed, and create vector DBs to query later.


Later when incorpoating citations for quotes, we will cite them based on chapter and percent of the way through the chapter (quote_start_char_position / num_chapter_chars)



Label the new books. 
 - add `verified` field in each dict in each json. If True, we've verified its classification, if False we haven't. 
 - The data labeling app, loads up all pages where `verified is False` with their initial classification. 
 - The display should be a horizontal scrolling list of pages. Where there is one page at center that we are classifying, and maybe one adjacent on each side taht is darkened to indicate that it is not under focus
 - We should be able to use the number keys to indicate which label applies. Once we set a label, set the `verified` field to True.
 - After every 10 labelings, we should save the current state of the dataset to a new tmp json file, then whenever we hit save, we should save the tmp file to the final json file.

Run training again.
