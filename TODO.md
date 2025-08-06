


From here we chunk everything up into sentences, paragraphs, and 200 words overlapping segments.
Each chunk should be saved with an embedding, the section its a part of, and position into the section of the start character of the chunk.
We should have databases for sentence embeddings, paragraph embeddings, and 200 word embeddings.

Following this, we should build a contextual database. Where each chunk will be pre-pended with the state of the story and characters up to that point, and this string (context + chunk) will be embedded.





Later when incorpoating citations for quotes, we will cite them based on chapter and percent of the way through the chapter (quote_start_char_position / num_chapter_chars)

