"""
Command-line interface for the Virtual Literature Companion system.

This module provides a comprehensive CLI for managing book ingestion, querying,
and system maintenance. It includes commands for:

- Book ingestion from PDF files
- Listing processed books
- System status and health checks
- Debug and maintenance operations
- Configuration management

The CLI is designed to be user-friendly with rich output formatting, progress
indicators, and comprehensive error handling.
"""

import click
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from .constants import DEBUG_MODE, BOOKS_DIR
from .ingest import load_ingest_file
from .llm.request import get_ai_status

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode with verbose logging')
@click.option('--quiet', is_flag=True, help='Suppress all output except errors')
@click.pass_context
def cli(ctx: click.Context, debug: bool, quiet: bool):
    """
    Virtual Literature Companion - Advanced book processing and analysis system.
    
    This tool processes books into structured, searchable formats with:
    - EPUB text extraction and section classification
    - Text cleaning using AI models
    - Section type detection (chapters, TOC, etc.)
    - Extensible architecture for multiple file formats
    
    Use 'vlc COMMAND --help' for detailed command information.
    """
    # Set up context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug or DEBUG_MODE
    ctx.obj['quiet'] = quiet
    
    # Configure logging based on flags
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Display system info in debug mode
    if debug:
        ai_status = get_ai_status()
        click.echo(f"Debug mode enabled")
        click.echo(f"Books directory: {BOOKS_DIR}")
        available_providers = [p for p, available in ai_status.get("providers", {}).items() if available]
        click.echo(f"Available AI providers: {', '.join(available_providers) if available_providers else 'none'}")


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, readable=True))
@click.option('--novel-name', '-n', required=True, help='Name of the novel')
@click.option('--author-name', '-a', required=True, help='Name of the author')
@click.pass_context
def ingest(
    ctx: click.Context,
    file_path: str,
    novel_name: str,
    author_name: str
):
    """
    Ingest a book file into the Virtual Literature Companion system.
    
    This command processes book files through the complete pipeline:
    
    \b
    1. File validation and format detection
    2. Section extraction and classification
    3. Text cleaning using AI models
    4. Structured data generation
    
    Currently supports:
    - EPUB files (.epub)
    
    Example:
        vlc ingest book.epub --novel-name "Pride and Prejudice" --author-name "Jane Austen"
    """
    if not ctx.obj['quiet']:
        click.echo(f"🔶 Virtual Literature Companion - Book Ingestion")
        click.echo(f"📚 Novel: {novel_name}")
        click.echo(f"👤 Author: {author_name}")
        click.echo(f"📄 File: {file_path}")
        click.echo("=" * 50)
    
    # Validate prerequisites
    ai_status = get_ai_status()
    providers_available = any(ai_status.get("providers", {}).values())
    if not providers_available:
        click.echo("⚠️  Warning: No AI providers available. Text cleaning will be limited.", err=True)
    
    # Start ingestion process
    try:
        # Convert file path to Path object
        file_path_obj = Path(file_path)
        
        click.echo(f"🔶 Processing {file_path_obj.name}...")
        
        # Process the file using the new ingest system
        sections = load_ingest_file(file_path_obj)
        
        # Create a simple result structure for now
        result = {
            'status': 'success',
            'novel_name': novel_name,
            'author_name': author_name,
            'sections': sections,
            'statistics': {
                'sections': len(sections),
                'total_characters': sum(len(s.get('clean_text', '')) for s in sections)
            }
        }
        
        # Display results
        _display_success_result(result, ctx.obj['quiet'])
            
    except KeyboardInterrupt:
        click.echo("\n❌ Processing interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {str(e)}", err=True)
        if ctx.obj['debug']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'summary']), default='table', help='Output format')
@click.option('--sort-by', '-s', type=click.Choice(['title', 'author', 'chapters', 'words']), default='title', help='Sort by field')
@click.pass_context
def list(ctx: click.Context, format: str, sort_by: str):
    """
    List all books that have been ingested into the system.
    
    This command displays information about all successfully processed books,
    including their metadata, processing statistics, and file locations.
    
    Output formats:
    - table: Human-readable table format (default)
    - json: Machine-readable JSON format
    - summary: Compact summary format
    
    Example:
        vlc list --format table --sort-by author
    """
    try:
        books = list_ingested_books()
        
        if not books:
            if not ctx.obj['quiet']:
                click.echo("📚 No books have been ingested yet.")
                click.echo("Use 'vlc ingest' to process your first book.")
            return
        
        # Sort books
        if sort_by == 'title':
            books.sort(key=lambda x: x.get('title', '').lower())
        elif sort_by == 'author':
            books.sort(key=lambda x: x.get('author_name', '').lower())
        elif sort_by == 'chapters':
            books.sort(key=lambda x: x.get('statistics', {}).get('total_chapters', 0), reverse=True)
        elif sort_by == 'words':
            books.sort(key=lambda x: x.get('statistics', {}).get('total_word_count', 0), reverse=True)
        
        # Display results
        if format == 'json':
            click.echo(json.dumps(books, indent=2, ensure_ascii=False))
        elif format == 'summary':
            _display_books_summary(books)
        else:  # table format
            _display_books_table(books)
            
    except Exception as e:
        click.echo(f"❌ Error listing books: {str(e)}", err=True)
        if ctx.obj['debug']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('novel_name')
@click.pass_context
def info(ctx: click.Context, novel_name: str):
    """
    Display detailed information about a specific book.
    
    This command shows comprehensive information about a processed book,
    including metadata, processing statistics, file locations, and
    available indexes.
    
    Example:
        vlc info "Pride and Prejudice"
    """
    try:
        books = list_ingested_books()
        book = None
        
        # Find the book (case-insensitive)
        for b in books:
            if b.get('title', '').lower() == novel_name.lower() or \
               b.get('directory_name', '').lower() == novel_name.lower():
                book = b
                break
        
        if not book:
            click.echo(f"❌ Book '{novel_name}' not found.")
            click.echo("Use 'vlc list' to see available books.")
            sys.exit(1)
        
        _display_book_info(book, ctx.obj['debug'])
        
    except Exception as e:
        click.echo(f"❌ Error getting book info: {str(e)}", err=True)
        if ctx.obj['debug']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """
    Display system status and health information.
    
    This command shows:
    - System configuration
    - Available dependencies
    - Storage usage
    - Processing statistics
    - API key status
    
    Example:
        vlc status
    """
    try:
        click.echo("🔶 Virtual Literature Companion - System Status")
        click.echo("=" * 50)
        
        # System configuration
        ai_status = get_ai_status()
        click.echo(f"📁 Books directory: {BOOKS_DIR}")
        click.echo(f"📊 Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
        available_providers = [p for p, available in ai_status.get("providers", {}).items() if available]
        click.echo(f"🤖 Available AI providers: {', '.join(available_providers) if available_providers else 'none'}")
        
        # Show model configurations if available
        if ai_status.get("text_clean_model", {}).get("provider"):
            provider = ai_status["text_clean_model"]["provider"]
            model = ai_status["text_clean_model"]["model"]
            click.echo(f"🔤 Text cleaning model: {model} ({provider})")
        
        # AI provider details
        click.echo(f"\n🔑 AI Configuration:")
        for provider, available in ai_status['providers'].items():
            status = "✅ Available" if available else "❌ Not available"
            click.echo(f"  {provider.title()}: {status}")
        
        # Check dependencies
        click.echo("\n📦 Dependencies:")
        dependencies = {
            'pdfplumber': 'PDF text extraction',
            'pytesseract': 'OCR processing',
            'chromadb': 'Vector database',
            'anthropic': 'Anthropic AI API',
            'openai': 'OpenAI API',
            'sentence_transformers': 'Text embeddings',
            'requests': 'Web scraping',
            'beautifulsoup4': 'HTML parsing',
            'python_dotenv': 'Environment configuration'
        }
        
        for dep, description in dependencies.items():
            try:
                __import__(dep)
                click.echo(f"  ✅ {dep}: {description}")
            except ImportError:
                click.echo(f"  ❌ {dep}: {description} (missing)")
        
        # Storage statistics
        books = list_ingested_books()
        total_books = len(books)
        total_chapters = sum(book.get('statistics', {}).get('total_chapters', 0) for book in books)
        total_words = sum(book.get('statistics', {}).get('total_word_count', 0) for book in books)
        
        click.echo(f"\n📈 Processing Statistics:")
        click.echo(f"  📚 Total books: {total_books}")
        click.echo(f"  📖 Total chapters: {total_chapters}")
        click.echo(f"  📝 Total words: {total_words:,}")
        
        # Storage usage
        if BOOKS_DIR.exists():
            total_size = sum(f.stat().st_size for f in BOOKS_DIR.rglob('*') if f.is_file())
            click.echo(f"  💾 Storage used: {_format_bytes(total_size)}")
        
        click.echo("\n✅ System operational")
        
    except Exception as e:
        click.echo(f"❌ Error checking system status: {str(e)}", err=True)
        if ctx.obj['debug']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('novel_name')
@click.confirmation_option(prompt='Are you sure you want to remove this book?')
@click.pass_context
def remove(ctx: click.Context, novel_name: str):
    """
    Remove a book and all its associated data.
    
    This command permanently deletes all files associated with a book,
    including raw text, structured data, metadata, and vector indexes.
    
    ⚠️  This action cannot be undone!
    
    Example:
        vlc remove "Pride and Prejudice"
    """
    try:
        books = list_ingested_books()
        book = None
        
        # Find the book
        for b in books:
            if b.get('title', '').lower() == novel_name.lower() or \
               b.get('directory_name', '').lower() == novel_name.lower():
                book = b
                break
        
        if not book:
            click.echo(f"❌ Book '{novel_name}' not found.")
            sys.exit(1)
        
        # Remove the book directory
        import shutil
        book_dir = Path(book['directory_path'])
        
        if book_dir.exists():
            shutil.rmtree(book_dir)
            click.echo(f"✅ Successfully removed '{book['title']}' by {book['author_name']}")
        else:
            click.echo(f"⚠️  Book directory not found: {book_dir}")
            
    except Exception as e:
        click.echo(f"❌ Error removing book: {str(e)}", err=True)
        if ctx.obj['debug']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# Helper functions for display formatting

def _display_success_result(result: Dict[str, Any], quiet: bool) -> None:
    """Display successful ingestion results."""
    if quiet:
        return
    
    click.echo("\n✅ Book ingestion completed successfully!")
    click.echo(f"📚 Title: {result['novel_name']}")
    click.echo(f"👤 Author: {result['author_name']}")
    
    stats = result.get('statistics', {})
    click.echo(f"📊 Statistics:")
    click.echo(f"  📖 Chapters: {stats.get('chapters', 0)}")
    click.echo(f"  📝 Paragraphs: {stats.get('paragraphs', 0)}")
    click.echo(f"  📄 Characters: {stats.get('total_characters', 0):,}")
    click.echo(f"  🔍 Indexes: {stats.get('indexes_created', 0)}")
    click.echo(f"  ⏱️  Processing time: {result.get('processing_time', 0):.2f} seconds")
    
    click.echo(f"\n📁 Output files:")
    for name, path in result.get('output_files', {}).items():
        click.echo(f"  {name}: {path}")


def _display_error_result(result: Dict[str, Any], debug: bool) -> None:
    """Display error results."""
    click.echo(f"❌ {result['message']}", err=True)
    
    error_details = result.get('error_details', {})
    if error_details:
        click.echo(f"Error type: {error_details.get('error_type', 'Unknown')}", err=True)
        click.echo(f"Processing step: {error_details.get('processing_step', 'Unknown')}", err=True)
        
        if debug and 'traceback' in error_details:
            click.echo("\nFull traceback:", err=True)
            click.echo(error_details['traceback'], err=True)


def _display_books_table(books: List[Dict[str, Any]]) -> None:
    """Display books in a formatted table."""
    if not books:
        return
    
    # Calculate column widths
    title_width = max(len(book.get('title', '')) for book in books)
    author_width = max(len(book.get('author_name', '')) for book in books)
    title_width = max(title_width, 20)
    author_width = max(author_width, 15)
    
    # Header
    click.echo(f"{'Title':<{title_width}} {'Author':<{author_width}} {'Chapters':<8} {'Words':<10} {'Indexes':<7}")
    click.echo("-" * (title_width + author_width + 33))
    
    # Rows
    for book in books:
        title = book.get('title', 'Unknown')[:title_width]
        author = book.get('author_name', 'Unknown')[:author_width]
        chapters = book.get('statistics', {}).get('total_chapters', 0)
        words = book.get('statistics', {}).get('total_word_count', 0)
        indexes = '✅' if book.get('has_indexes', False) else '❌'
        
        click.echo(f"{title:<{title_width}} {author:<{author_width}} {chapters:<8} {words:<10,} {indexes:<7}")
    
    click.echo(f"\nTotal: {len(books)} book(s)")


def _display_books_summary(books: List[Dict[str, Any]]) -> None:
    """Display books in a compact summary format."""
    for book in books:
        title = book.get('title', 'Unknown')
        author = book.get('author_name', 'Unknown')
        chapters = book.get('statistics', {}).get('total_chapters', 0)
        words = book.get('statistics', {}).get('total_word_count', 0)
        
        click.echo(f"📚 {title} by {author} ({chapters} chapters, {words:,} words)")


def _display_book_info(book: Dict[str, Any], debug: bool) -> None:
    """Display detailed information about a specific book."""
    click.echo(f"📚 {book.get('title', 'Unknown Title')}")
    click.echo(f"👤 Author: {book.get('author_name', 'Unknown Author')}")
    click.echo(f"📅 Publication: {book.get('publication_date', 'Unknown')}")
    click.echo(f"📁 Directory: {book.get('directory_path', 'Unknown')}")
    
    # Statistics
    stats = book.get('statistics', {})
    click.echo(f"\n📊 Statistics:")
    click.echo(f"  📖 Chapters: {stats.get('total_chapters', 0)}")
    click.echo(f"  📝 Paragraphs: {stats.get('total_paragraphs', 0)}")
    click.echo(f"  📄 Words: {stats.get('total_word_count', 0):,}")
    click.echo(f"  👥 Characters: {stats.get('unique_characters', 0)}")
    click.echo(f"  🏞️  Settings: {stats.get('unique_settings', 0)}")
    
    # Author bio
    author_bio = book.get('author_bio', '')
    if author_bio and len(author_bio) > 50:
        click.echo(f"\n📝 Author Bio:")
        click.echo(f"  {author_bio[:200]}...")
    
    # Table of contents
    toc = book.get('table_of_contents', [])
    if toc:
        click.echo(f"\n📑 Table of Contents:")
        for chapter in toc[:10]:  # Show first 10 chapters
            title = chapter.get('chapter_title', f"Chapter {chapter.get('chapter_num', '?')}")
            words = chapter.get('word_count', 0)
            click.echo(f"  {chapter.get('chapter_num', '?'):2}. {title} ({words:,} words)")
        
        if len(toc) > 10:
            click.echo(f"  ... and {len(toc) - 10} more chapters")
    
    # Indexes
    click.echo(f"\n🔍 Indexes: {'Available' if book.get('has_indexes', False) else 'Not available'}")
    
    if debug:
        click.echo(f"\n🔧 Debug Information:")
        click.echo(f"  Processing version: {book.get('processing_metadata', {}).get('version', 'Unknown')}")
        click.echo(f"  Full directory: {book.get('directory_path', 'Unknown')}")


def _format_bytes(bytes_count: int) -> str:
    """Format bytes count into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} TB"


if __name__ == '__main__':
    cli() 