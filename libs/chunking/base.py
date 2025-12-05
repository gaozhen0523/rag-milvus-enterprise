from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """
    Base class for text chunking strategy.
    Different chunk strategies (by words, sentences, tokens) should inherit this.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """
        Splits text into chunks.
        """
        pass
