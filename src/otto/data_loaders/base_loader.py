from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional

from otto.utils.documents.document_utils import Blob, Document
from otto.utils.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

_HAS_TEXT_SPLITTERS = True

class BaseLoader(ABC):
    def load(self) -> list[Document]:
        return list(self.lazy_load())
    
    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> list[Document]:
        """Load Documents and split into chunks. Chunks are returned as Documents.

        Do not override this method. It should be considered to be deprecated!

        Args:
            text_splitter: TextSplitter instance to use for splitting documents.
                Defaults to RecursiveCharacterTextSplitter.

        Raises:
            ImportError: If langchain-text-splitters is not installed
                and no text_splitter is provided.

        Returns:
            List of Documents.
        """
        if text_splitter is None:
            if not _HAS_TEXT_SPLITTERS:
                msg = (
                    "Unable to import from langchain_text_splitters. Please specify "
                    "text_splitter or install langchain_text_splitters with "
                    "`pip install -U langchain-text-splitters`."
                )
                raise ImportError(msg)

            text_splitter_: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            text_splitter_ = text_splitter
        docs = self.load()
        return text_splitter_.split_documents(docs)
    
    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        if type(self).load != BaseLoader.load:
            return iter(self.load())
        msg = f"{self.__class__.__name__} does not implement lazy_load()"
        raise NotImplementedError(msg)
    
class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    A blob parser provides a way to parse raw data stored in a blob into one
    or more documents.

    The parser can be composed with blob loaders, making it easy to reuse
    a parser independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface.

        Subclasses are required to implement this method.

        Args:
            blob: Blob instance

        Returns:
            Generator of documents
        """

    def parse(self, blob: Blob) -> list[Document]:
        """Eagerly parse the blob into a document or documents.

        This is a convenience method for interactive development environment.

        Production applications should favor the lazy_parse method instead.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: Blob instance

        Returns:
            List of documents
        """
        return list(self.lazy_parse(blob))
