"""
PDF and document extractor for collecting text data from PDFs.
Handles various PDF formats and extracts structured text.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extract text from PDF documents.
    
    Features:
    - Text extraction from digital PDFs
    - OCR capability for scanned documents
    - Metadata preservation
    - Batch processing
    """
    
    def __init__(self, use_ocr: bool = False):
        """
        Initialize PDF extractor.
        
        Args:
            use_ocr: Enable OCR for scanned documents
        """
        self.use_ocr = use_ocr
        
    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted documents (one per page or chunk)
        """
        documents = []
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return documents
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    # Use OCR if text extraction fails and OCR is enabled
                    if not text and self.use_ocr:
                        text = self._extract_with_ocr(page)
                    
                    if text:
                        documents.append({
                            'content': text,
                            'source_type': 'pdf',
                            'source_file': str(pdf_path),
                            'page': page_num + 1,
                            'metadata': {
                                'title': pdf.metadata.get('Title', ''),
                                'author': pdf.metadata.get('Author', ''),
                                'total_pages': len(pdf.pages)
                            }
                        })
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {e}")
            
        return documents
    
    def extract_from_directory(self, 
                              directory: str, 
                              recursive: bool = True) -> List[Dict[str, str]]:
        """
        Extract text from all PDFs in a directory.
        
        Args:
            directory: Directory path containing PDFs
            recursive: Whether to search subdirectories
            
        Returns:
            List of all extracted documents
        """
        documents = []
        path = Path(directory)
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        
        for pdf_file in path.glob(pattern):
            logger.info(f"Extracting from {pdf_file}")
            docs = self.extract_from_pdf(str(pdf_file))
            documents.extend(docs)
            
        return documents
    
    def _extract_with_ocr(self, page) -> Optional[str]:
        """Extract text from page using OCR."""
        try:
            image = page.to_image()
            text = pytesseract.image_to_string(image.original)
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = PDFExtractor(use_ocr=False)
    # docs = extractor.extract_from_directory("./pdf_data/")
    # print(f"Extracted {len(docs)} documents")
