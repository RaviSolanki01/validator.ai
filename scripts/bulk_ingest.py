#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
from typing import List

from core.ingestion.preprocessor import DocumentPreprocessor
from core.ingestion.chunking import DocumentChunker
from infrastructure.vector_db.client import VectorDBClient
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BulkIngestor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.preprocessor = DocumentPreprocessor()
        self.chunker = DocumentChunker()
        self.vector_db = VectorDBClient(settings)

    def process_document(self, file_path: Path) -> bool:
        try:
            # Preprocess the document
            logger.info(f"Preprocessing document: {file_path}")
            processed_doc = self.preprocessor.process(file_path)

            # Chunk the processed document
            logger.info(f"Chunking document: {file_path}")
            chunks = self.chunker.chunk(processed_doc)

            # Store chunks in vector database
            logger.info(f"Storing chunks in vector database: {file_path}")
            self.vector_db.store_chunks(chunks, metadata={"source": str(file_path)})

            logger.info(f"Successfully processed and stored: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False

    def process_directory(self, directory_path: Path) -> tuple[int, int]:
        """Process all documents in a directory.

        Args:
            directory_path: Path to the directory containing documents

        Returns:
            tuple: (number of successful ingestions, total number of documents)
        """
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")

        supported_extensions = [".pdf", ".txt", ".doc", ".docx"]
        documents = [
            f for f in directory_path.rglob("*")
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        total_docs = len(documents)
        if total_docs == 0:
            logger.warning(f"No supported documents found in {directory_path}")
            return 0, 0

        successful = 0
        for doc in documents:
            if self.process_document(doc):
                successful += 1

        return successful, total_docs

def main():
    parser = argparse.ArgumentParser(description="Bulk ingest documents into the system")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing documents to ingest"
    )
    args = parser.parse_args()

    settings = Settings()
    ingestor = BulkIngestor(settings)

    try:
        directory_path = Path(args.directory)
        successful, total = ingestor.process_directory(directory_path)
        
        logger.info(f"Ingestion complete: {successful}/{total} documents processed successfully")
        if successful < total:
            logger.warning(f"Failed to process {total - successful} documents")
            return 1
        return 0

    except Exception as e:
        logger.error(f"Fatal error during ingestion: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())