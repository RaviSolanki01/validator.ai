from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import uuid4

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentType(Enum):
    """Supported document types for preprocessing."""
    TEXT = 'text'
    HTML = 'html'
    PDF = 'pdf'
    DOCX = 'docx'
    CSV = 'csv'
    EXCEL = 'excel'
    JSON = 'json'
    XML = 'xml'

@dataclass
class DocumentMetadata:
    """Metadata for preprocessed documents."""
    doc_id: str
    doc_type: DocumentType
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    language: Optional[str] = None
    source_path: Optional[Path] = None
    word_count: int = 0
    sentence_count: int = 0
    custom_metadata: Dict = field(default_factory=dict)

@dataclass
class PreprocessedDocument:
    """Container for preprocessed document content and metadata."""
    content: str
    metadata: DocumentMetadata
    sections: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

class DocumentPreprocessor:
    """Handles document preprocessing tasks including cleaning, normalization, and metadata extraction."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the preprocessor with optional configuration.

        Args:
            config: Configuration dictionary for customizing preprocessing behavior
        """
        self.config = config or {}
        self._initialize_nltk()

    def _initialize_nltk(self) -> None:
        """Initialize NLTK resources if not already downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')

    def preprocess(self, content: Union[str, Path], doc_type: DocumentType = None) -> PreprocessedDocument:
        """Preprocess the document content and extract metadata.

        Args:
            content: Raw document content or path to document
            doc_type: Type of document being processed

        Returns:
            PreprocessedDocument object containing processed content and metadata
        """
        if isinstance(content, Path):
            doc_type = doc_type or self._detect_doc_type(content)
            content = self._load_document(content, doc_type)

        # Clean and normalize content
        cleaned_content = self._clean_text(content, doc_type)
        normalized_content = self._normalize_text(cleaned_content)

        # Extract metadata
        metadata = self._extract_metadata(normalized_content, doc_type)

        # Process content further
        sections = self._extract_sections(normalized_content)
        keywords = self._extract_keywords(normalized_content)
        summary = self._generate_summary(normalized_content)

        return PreprocessedDocument(
            content=normalized_content,
            metadata=metadata,
            sections=sections,
            keywords=keywords,
            summary=summary
        )

    def _detect_doc_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file extension."""
        extension = file_path.suffix.lower()
        extension_map = {
            '.txt': DocumentType.TEXT,
            '.html': DocumentType.HTML,
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.csv': DocumentType.CSV,
            '.xlsx': DocumentType.EXCEL,
            '.json': DocumentType.JSON,
            '.xml': DocumentType.XML
        }
        return extension_map.get(extension, DocumentType.TEXT)

    def _load_document(self, file_path: Path, doc_type: DocumentType) -> str:
        """Load document content based on document type."""
        if doc_type == DocumentType.HTML:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text()
        elif doc_type in [DocumentType.CSV, DocumentType.EXCEL]:
            df = pd.read_csv(file_path) if doc_type == DocumentType.CSV else pd.read_excel(file_path)
            return df.to_string()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _clean_text(self, text: str, doc_type: DocumentType) -> str:
        """Clean text by removing unwanted characters and formatting."""
        if doc_type == DocumentType.HTML:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters while preserving essential punctuation
        text = text.replace('\t', ' ')
        text = text.replace('\r', '\n')

        return text

    def _normalize_text(self, text: str) -> str:
        """Normalize text by standardizing formatting and encoding."""
        # Convert to lowercase if specified in config
        if self.config.get('lowercase', False):
            text = text.lower()

        # Normalize quotes and dashes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u2013', '-').replace('\u2014', '--')

        # Normalize whitespace
        text = text.replace('\n\n+', '\n\n')

        return text

    def _extract_metadata(self, text: str, doc_type: DocumentType) -> DocumentMetadata:
        """Extract metadata from document content."""
        # Generate unique document ID
        doc_id = str(uuid4())

        # Detect language
        try:
            language = detect(text[:1000])  # Use first 1000 chars for language detection
        except:
            language = None

        # Count words and sentences
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        return DocumentMetadata(
            doc_id=doc_id,
            doc_type=doc_type,
            language=language,
            word_count=len(words),
            sentence_count=len(sentences)
        )

    def _extract_sections(self, text: str) -> List[str]:
        """Extract logical sections from the document."""
        # Split text into sections based on double newlines or headings
        sections = [s.strip() for s in text.split('\n\n') if s.strip()]

        # Filter out very short sections
        min_section_length = self.config.get('min_section_length', 50)
        sections = [s for s in sections if len(s) >= min_section_length]

        return sections

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the document."""
        # Tokenize and tag parts of speech
        words = word_tokenize(text.lower())
        tagged = nltk.pos_tag(words)

        # Extract nouns and proper nouns
        keywords = [word for word, tag in tagged if tag.startswith(('NN', 'NNP'))]

        # Count frequency and get top keywords
        from collections import Counter
        keyword_freq = Counter(keywords)
        num_keywords = self.config.get('num_keywords', 10)

        return [word for word, _ in keyword_freq.most_common(num_keywords)]

    def _generate_summary(self, text: str) -> Optional[str]:
        """Generate a brief summary of the document."""
        # Split into sentences
        sentences = sent_tokenize(text)

        # For now, use first few sentences as summary
        num_summary_sentences = self.config.get('num_summary_sentences', 3)
        if len(sentences) <= num_summary_sentences:
            return text

        return ' '.join(sentences[:num_summary_sentences])