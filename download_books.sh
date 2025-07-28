#!/bin/bash

# Change to the target directory
cd ~/Desktop/virtual_literature_companion/book_source/

echo "Downloading public domain books..."

# Function to download and rename files
download_book() {
    local url="$1"
    local filename="$2"
    local format="$3"
    
    echo "Downloading: $filename.$format"
    
    if command -v wget &> /dev/null; then
        wget -q -O "${filename}.${format}" "$url"
    elif command -v curl &> /dev/null; then
        curl -s -L -o "${filename}.${format}" "$url"
    else
        echo "Error: Neither wget nor curl is available"
        return 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ Downloaded: ${filename}.${format}"
    else
        echo "✗ Failed to download: ${filename}.${format}"
    fi
}

# Download books from Project Gutenberg as EPUBs

# Classic Novels
download_book "https://www.gutenberg.org/ebooks/2600.epub.noimages" "war_and_peace" "epub"
download_book "https://www.gutenberg.org/ebooks/135.epub.noimages" "les_misérables" "epub"

# Philosophy & Essays  
download_book "https://www.gutenberg.org/ebooks/3600.epub.noimages" "the_complete_essays" "epub"
download_book "https://www.gutenberg.org/ebooks/1497.epub.noimages" "the_republic" "epub"
download_book "https://www.gutenberg.org/ebooks/2680.epub.noimages" "meditations" "epub"

# Poetry & Drama
download_book "https://www.gutenberg.org/ebooks/100.epub.noimages" "the_complete_works_of_william_shakespeare" "epub"
download_book "https://www.gutenberg.org/ebooks/8800.epub.noimages" "the_divine_comedy" "epub"
download_book "https://www.gutenberg.org/ebooks/1322.epub.noimages" "leaves_of_grass" "epub"

# Historical & Religious Texts
download_book "https://www.gutenberg.org/ebooks/731.epub.noimages" "the_decline_and_fall_of_the_roman_empire_vol_1" "epub"
download_book "https://www.gutenberg.org/ebooks/732.epub.noimages" "the_decline_and_fall_of_the_roman_empire_vol_2" "epub"
download_book "https://www.gutenberg.org/ebooks/10.epub.noimages" "the_king_james_bible" "epub"

# Adventure & Gothic Literature
download_book "https://www.gutenberg.org/ebooks/1184.epub.noimages" "the_count_of_monte_cristo" "epub"
download_book "https://www.gutenberg.org/ebooks/84.epub.noimages" "frankenstein" "epub"
download_book "https://www.gutenberg.org/ebooks/76.epub.noimages" "the_adventures_of_huckleberry_finn" "epub"

# Science & Natural Philosophy
download_book "https://www.gutenberg.org/ebooks/1228.epub.noimages" "on_the_origin_of_species" "epub"
download_book "https://www.gutenberg.org/ebooks/944.epub.noimages" "the_voyage_of_the_beagle" "epub"

# Non-English Originals
download_book "https://www.gutenberg.org/ebooks/996.epub.noimages" "don_quixote" "epub"
download_book "https://www.gutenberg.org/ebooks/26155.epub.noimages" "the_tale_of_genji" "epub"

# Autobiography & Memoirs
download_book "https://www.gutenberg.org/ebooks/20203.epub.noimages" "the_autobiography_of_benjamin_franklin" "epub"
download_book "https://www.gutenberg.org/ebooks/3913.epub.noimages" "confessions" "epub"
download_book "https://www.gutenberg.org/ebooks/2376.epub.noimages" "up_from_slavery" "epub"

echo ""
echo "Download complete! All books saved as EPUBs to: ~/Desktop/virtual_literature_companion/book_source/"
echo ""
echo "All files are in EPUB format and ready to use!"

echo ""
echo "All downloads finished!"
