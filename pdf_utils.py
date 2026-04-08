"""PDF text extraction utilities for insurance claim documents."""

from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text content from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        Exception: If the PDF cannot be read.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()
