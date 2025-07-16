#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer

from config.settings import Settings

class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    start_index: int
    end_index: int
    chunk_type: str
    overlap_previous: bool
    overlap_next: bool
    custom_metadata: Dict[str, Any]

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    text: str
    metadata: ChunkMetadata

class DocumentChunker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def chunk(self, text: str, strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
              chunk_size: int = 512, overlap: int = 50) -> List[DocumentChunk]:
        """Split text into chunks using the specified strategy.

        Args:
            text: The text to chunk
            strategy: Chunking strategy to use
            chunk_size: Maximum size of each chunk (in tokens)
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of document chunks
        """
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(text, chunk_size, overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _fixed_size_chunking(self, text: str, chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Split text into fixed-size chunks with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            # Calculate end position with overlap
            end = min(start + chunk_size, len(tokens))
            
            # Decode chunk back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Create chunk metadata
            metadata = ChunkMetadata(
                start_index=start,
                end_index=end,
                chunk_type="fixed_size",
                overlap_previous=start > 0,
                overlap_next=end < len(tokens),
                custom_metadata={}
            )
            
            chunks.append(DocumentChunk(chunk_text, metadata))
            
            # Move start position, accounting for overlap
            start = end - overlap if end < len(tokens) else len(tokens)
        
        return chunks

    def _sentence_chunking(self, text: str, chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Split text into chunks at sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence)
            sentence_length = len(sentence_tokens)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = " ".join(current_chunk)
                metadata = ChunkMetadata(
                    start_index=0,  # Placeholder
                    end_index=0,    # Placeholder
                    chunk_type="sentence",
                    overlap_previous=len(chunks) > 0,
                    overlap_next=True,
                    custom_metadata={"num_sentences": len(current_chunk)}
                )
                chunks.append(DocumentChunk(chunk_text, metadata))
                
                # Start new chunk, keeping overlap
                overlap_sentences = current_chunk[-2:] if overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(self.tokenizer.encode(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk if there are remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            metadata = ChunkMetadata(
                start_index=0,  # Placeholder
                end_index=0,    # Placeholder
                chunk_type="sentence",
                overlap_previous=len(chunks) > 0,
                overlap_next=False,
                custom_metadata={"num_sentences": len(current_chunk)}
            )
            chunks.append(DocumentChunk(chunk_text, metadata))
        
        return chunks

    def _paragraph_chunking(self, text: str, chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Split text into chunks at paragraph boundaries."""
        # Split text into paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            para_tokens = self.tokenizer.encode(paragraph)
            para_length = len(para_tokens)
            
            if current_length + para_length > chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = "\n\n".join(current_chunk)
                metadata = ChunkMetadata(
                    start_index=0,  # Placeholder
                    end_index=0,    # Placeholder
                    chunk_type="paragraph",
                    overlap_previous=len(chunks) > 0,
                    overlap_next=True,
                    custom_metadata={"num_paragraphs": len(current_chunk)}
                )
                chunks.append(DocumentChunk(chunk_text, metadata))
                
                # Start new chunk, keeping overlap
                overlap_paras = current_chunk[-1:] if overlap > 0 else []
                current_chunk = overlap_paras + [paragraph]
                current_length = sum(len(self.tokenizer.encode(p)) for p in current_chunk)
            else:
                current_chunk.append(paragraph)
                current_length += para_length
        
        # Add final chunk if there are remaining paragraphs
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            metadata = ChunkMetadata(
                start_index=0,  # Placeholder
                end_index=0,    # Placeholder
                chunk_type="paragraph",
                overlap_previous=len(chunks) > 0,
                overlap_next=False,
                custom_metadata={"num_paragraphs": len(current_chunk)}
            )
            chunks.append(DocumentChunk(chunk_text, metadata))
        
        return chunks

    def _semantic_chunking(self, text: str, chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Split text into chunks based on semantic boundaries.
        
        This method combines sentence and paragraph chunking with additional
        semantic analysis to create more meaningful chunks.
        """
        # First, split into paragraph-based chunks
        initial_chunks = self._paragraph_chunking(text, chunk_size * 2, overlap)
        final_chunks = []
        
        for chunk in initial_chunks:
            # Further split large paragraphs into semantic units
            if len(self.tokenizer.encode(chunk.text)) > chunk_size:
                sentences = sent_tokenize(chunk.text)
                current_group = []
                current_length = 0
                
                for sentence in sentences:
                    sent_tokens = self.tokenizer.encode(sentence)
                    sent_length = len(sent_tokens)
                    
                    if current_length + sent_length > chunk_size and current_group:
                        # Create semantic chunk
                        semantic_text = " ".join(current_group)
                        metadata = ChunkMetadata(
                            start_index=0,  # Placeholder
                            end_index=0,    # Placeholder
                            chunk_type="semantic",
                            overlap_previous=len(final_chunks) > 0,
                            overlap_next=True,
                            custom_metadata={
                                "original_paragraph": True,
                                "num_sentences": len(current_group)
                            }
                        )
                        final_chunks.append(DocumentChunk(semantic_text, metadata))
                        
                        # Start new group with overlap
                        current_group = current_group[-1:] + [sentence] if overlap > 0 else [sentence]
                        current_length = sum(len(self.tokenizer.encode(s)) for s in current_group)
                    else:
                        current_group.append(sentence)
                        current_length += sent_length
                
                # Add final semantic group
                if current_group:
                    semantic_text = " ".join(current_group)
                    metadata = ChunkMetadata(
                        start_index=0,  # Placeholder
                        end_index=0,    # Placeholder
                        chunk_type="semantic",
                        overlap_previous=len(final_chunks) > 0,
                        overlap_next=False,
                        custom_metadata={
                            "original_paragraph": True,
                            "num_sentences": len(current_group)
                        }
                    )
                    final_chunks.append(DocumentChunk(semantic_text, metadata))
            else:
                # Keep small paragraphs as is
                chunk.metadata.chunk_type = "semantic"
                final_chunks.append(chunk)
        
        return final_chunks