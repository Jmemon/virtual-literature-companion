



From here we chunk (keeping track of chunk positions), embed, and create vector DBs to query later.


Later when incorpoating citations for quotes, we will cite them based on chapter and percent of the way through the chapter (quote_start_char_position / num_chapter_chars)



Label the new books. 
 - add `verified` field in each dict in each json. If True, we've verified its classification, if False we haven't. 
 - The data labeling app, loads up all pages where `verified is False` with their initial classification. 
 - The display should be a horizontal scrolling list of pages. Where there is one page at center that we are classifying, and maybe one adjacent on each side taht is darkened to indicate that it is not under focus
 - We should be able to use the number keys to indicate which label applies. Once we set a label, set the `verified` field to True.
 - After every 10 labelings, we should save the current state of the dataset to a new tmp json file, then whenever we hit save, we should save the tmp file to the final json file.

Run training again.
