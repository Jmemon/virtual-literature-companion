# Virtual Literature Companion

**Advanced PDF book processing and analysis system with AI-powered literary analysis**

The Virtual Literature Companion is a comprehensive system for converting PDF books into structured, searchable, and analyzable formats. It uses advanced AI and machine learning techniques to understand literature at a deep level, making it perfect for researchers, students, and book enthusiasts.

## 🚀 Key Features

### 📚 **Complete Book Processing Pipeline**
- **Multi-threaded PDF text extraction** with OCR fallback for scanned books
- **Intelligent chapter detection** using multiple pattern recognition strategies
- **Robust text structuring** with paragraph-level analysis
- **Error recovery** and partial result cleanup

### 🧠 **AI-Powered Literary Analysis**
- **Character identification** and tracking across chapters
- **Setting detection** and scene localization
- **Dialogue classification** and speech pattern analysis
- **OpenAI GPT integration** for sophisticated literary understanding

### 🔍 **Advanced Search Capabilities**
- **Vector embeddings** using state-of-the-art sentence transformers
- **Dual indexing system**: siloed (content-only) and contextual (character/setting enhanced)
- **Semantic search** that understands meaning, not just keywords
- **ChromaDB integration** for persistent, scalable storage

### 📊 **Rich Metadata Generation**
- **Author biographies** from web sources
- **Publication information** and historical context
- **Comprehensive statistics** (word counts, character analysis, etc.)
- **Table of contents** with chapter summaries

### 🛠️ **Production-Ready CLI**
- **Beautiful command-line interface** with progress tracking
- **Comprehensive error handling** and user feedback
- **Multiple output formats** (table, JSON, summary)
- **Book management** (list, info, remove)

## 🏗️ System Architecture

```
Virtual Literature Companion
├── 📄 PDF Input
│   ├── Text-based PDFs (direct extraction)
│   ├── Scanned PDFs (OCR processing)
│   └── Mixed PDFs (hybrid approach)
│
├── 🔧 Processing Pipeline
│   ├── 1. PDF to Text (multi-threaded)
│   ├── 2. Chapter Detection & Splitting
│   ├── 3. Paragraph Analysis (AI-powered)
│   ├── 4. Metadata Generation
│   └── 5. Vector Index Creation
│
├── 💾 Structured Output
│   ├── Raw text files (per chapter)
│   ├── Structured JSON (with analysis)
│   ├── Book metadata (author, stats)
│   └── Vector indexes (dual-type)
│
└── 🔍 Query Interface
    ├── Semantic search
    ├── Character-based queries
    ├── Setting-based searches
    └── Contextual analysis
```

## 📦 Installation

### Prerequisites
- **Python 3.11+** (required for modern typing and async features)
- **OpenAI API key** (optional but recommended for full literary analysis)
- **Tesseract OCR** (for scanned PDF processing)

### Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Install the System

1. **Clone the repository:**
```bash
git clone <repository-url>
cd virtual-literature-companion
```

2. **Install dependencies:**
```bash
pip install -e .
```

3. **Set up OpenAI API key (optional):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 🚀 Quick Start

### 1. **Ingest Your First Book**
```bash
vlc ingest book.pdf --novel-name "Pride and Prejudice" --author-name "Jane Austen"
```

### 2. **List Processed Books**
```bash
vlc list
```

### 3. **Get Detailed Book Information**
```bash
vlc info "Pride and Prejudice"
```

### 4. **Check System Status**
```bash
vlc status
```

## 📖 Detailed Usage Guide

### **Book Ingestion**

The ingestion process transforms a PDF book through multiple stages:

```bash
# Basic ingestion
vlc ingest my_book.pdf -n "Novel Name" -a "Author Name"

# Advanced options
vlc ingest my_book.pdf \
  --novel-name "1984" \
  --author-name "George Orwell" \
  --skip-existing \
  --no-cleanup \
  --embedding-model "all-MiniLM-L6-v2"
```

**Options:**
- `--skip-existing`: Skip if book already processed
- `--no-cleanup`: Don't clean up partial results on error
- `--embedding-model`: Choose embedding model for search

### **Book Management**

```bash
# List all books with statistics
vlc list --format table --sort-by author

# JSON output for programmatic use
vlc list --format json

# Compact summary
vlc list --format summary

# Detailed book information
vlc info "Book Title"

# Remove a book and all its data
vlc remove "Book Title"
```

### **System Monitoring**

```bash
# Check system health
vlc status

# Debug mode for troubleshooting
vlc --debug ingest book.pdf -n "Title" -a "Author"
```

## 🏗️ File Structure

After processing, each book creates the following structure:

```
books/
└── Novel_Name/
    ├── raw_chapters/          # Raw text files per chapter
    │   ├── 1.txt
    │   ├── 2.txt
    │   └── ...
    ├── structured/            # Analyzed JSON data
    │   ├── 1.json
    │   ├── 2.json
    │   └── ...
    ├── indexes/               # Vector databases
    │   ├── chroma.sqlite3
    │   └── ...
    └── metadata.json          # Book metadata
```

### **JSON Structure Example**

```json
{
  "chapter_num": 1,
  "chapter_title": "Chapter 1: The Beginning",
  "paragraphs": [
    {
      "characters": ["Winston Smith", "O'Brien"],
      "setting": ["Victory Mansions", "London"],
      "paragraph_idx": 0,
      "text_path": "books/1984/raw_chapters/1.txt",
      "word_count": 45,
      "dialogue": false,
      "embedding_id": "uuid-here"
    }
  ]
}
```

## 🔧 Advanced Configuration

### **Environment Variables**
```bash
# Core configuration
export OPENAI_API_KEY="your-key"
export DEBUG_MODE="false"
export LOG_LEVEL="INFO"

# Processing tuning
export MAX_WORKERS="4"
export TESSERACT_CONFIG="--psm 6"
```

### **Customizing Models**
```python
from virtual_literature_companion.constants import DEFAULT_EMBEDDING_MODEL

# Available embedding models
models = [
    "all-MiniLM-L6-v2",        # Fast, good quality
    "all-mpnet-base-v2",       # Higher quality, slower
    "multi-qa-MiniLM-L6-cos-v1" # Optimized for Q&A
]
```

## 🔍 Search and Analysis

### **Programmatic Usage**
```python
from virtual_literature_companion import ingest_book_pdf, list_ingested_books

# Process a book
result = ingest_book_pdf(
    pdf_path="book.pdf",
    novel_name="1984",
    author_name="George Orwell"
)

# List processed books
books = list_ingested_books()
```

### **Vector Search Example**
```python
from virtual_literature_companion.processors.create_vector_indexes import VectorIndexCreator

# Create index creator
creator = VectorIndexCreator("1984")

# Query the contextual index
results = creator.test_index_query(
    "1984_contextual",
    "Winston's relationship with Big Brother",
    n_results=5
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_virtual_literature_companion.py

# Run specific test categories
python -m unittest test_virtual_literature_companion.TestPDFProcessing
python -m unittest test_virtual_literature_companion.TestLiteraryAnalysis
python -m unittest test_virtual_literature_companion.TestVectorIndexing
```

**Test Coverage:**
- ✅ PDF processing and text extraction
- ✅ Chapter detection and splitting
- ✅ Literary analysis and character identification
- ✅ Vector indexing and search
- ✅ Complete ingestion pipeline
- ✅ CLI interface functionality
- ✅ Error handling and edge cases

## 🔧 Technical Details

### **PDF Processing Trade-offs**
- **Text extraction vs OCR**: We try text extraction first (fast, accurate) and fall back to OCR (slower, good for scanned books)
- **Multi-threading**: Pages are processed in parallel for speed, but reassembled in order
- **Memory management**: Large PDFs are processed in chunks to avoid memory issues

### **AI Analysis Considerations**
- **Token limits**: Paragraphs are analyzed individually to stay within API limits
- **Cost optimization**: We use the fastest model (gpt-4o-mini) for analysis
- **Fallback gracefully**: If OpenAI isn't available, we use rule-based analysis

### **Vector Search Design**
- **Dual indexing**: Siloed (pure content) and contextual (character+setting enhanced)
- **Embedding model choice**: all-MiniLM-L6-v2 provides good speed/quality balance
- **Persistent storage**: ChromaDB ensures indexes survive between sessions

## 🤝 Contributing

### **Development Setup**
```bash
# Clone and install in development mode
git clone <repository-url>
cd virtual-literature-companion
pip install -e ".[dev]"

# Run tests
python test_virtual_literature_companion.py
```

### **Code Style**
- **Type hints**: All functions have comprehensive type annotations
- **Docstrings**: Every function has detailed docstrings explaining purpose and usage
- **Error handling**: Robust error handling with informative messages
- **Logging**: Comprehensive logging for debugging and monitoring

## 📊 Performance Characteristics

### **Processing Speed**
- **Small book** (100 pages): ~2-5 minutes
- **Medium book** (300 pages): ~5-15 minutes
- **Large book** (500+ pages): ~15-30 minutes

### **Storage Requirements**
- **Raw text**: ~1-2MB per 100 pages
- **Structured JSON**: ~2-5MB per 100 pages
- **Vector indexes**: ~5-10MB per 100 pages

### **Memory Usage**
- **PDF processing**: ~50-100MB per worker thread
- **Vector indexing**: ~100-200MB during creation
- **Normal operation**: ~20-50MB

## 🚨 Troubleshooting

### **Common Issues**

**1. OCR not working**
```bash
# Check if tesseract is installed
tesseract --version

# On macOS
brew install tesseract

# On Ubuntu
sudo apt-get install tesseract-ocr
```

**2. OpenAI API errors**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test with curl
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

**3. Memory issues with large PDFs**
```bash
# Reduce worker threads
export MAX_WORKERS="2"

# Enable debug mode
vlc --debug ingest book.pdf -n "Title" -a "Author"
```

### **Debug Mode**
```bash
# Enable comprehensive logging
export DEBUG_MODE="true"
vlc --debug ingest book.pdf -n "Title" -a "Author"
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT models for literary analysis
- **Hugging Face** for the sentence transformer models
- **ChromaDB** for the vector database infrastructure
- **pdfplumber** and **PyMuPDF** for PDF processing capabilities
- **Tesseract** for OCR functionality

---

**Ready to transform your PDF library into a searchable, analyzable literary database?**

Get started with: `vlc ingest your_book.pdf --novel-name "Title" --author-name "Author"`
