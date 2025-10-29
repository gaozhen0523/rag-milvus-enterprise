from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """
    Base class for text chunking strategy.
    Different chunk strategies (by words, sentences, tokens) should inherit this.
    """

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Splits text into chunks.
        """
        pass