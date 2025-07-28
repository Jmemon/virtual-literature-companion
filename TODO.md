

Ingest the new books into jsons. Set up book ingestion so that stages can be run independently. In this instance so that we can extract the pages from the new epubs and save the page-by-page jsons to `page_classifier_sidequest/page_dataset/raw` as its own operation. Then we will re-label them using the labeling app. Then we will will run training with this far larger dataset.

Label them. 

Run training again.