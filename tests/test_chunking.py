# tests/test_chunking.py
import pytest

from libs.chunking.text_chunker import TextChunker


def test_char_strategy_basic():
    txt = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    chunker = TextChunker(strategy="char", size=10, overlap=2)
    chunks = chunker.chunk(txt)
    # 期望窗口: [0:10], [8:18], [16:26]
    assert [c.start for c in chunks] == [0, 8, 16]
    assert [c.end for c in chunks] == [10, 18, 26]
    assert chunks[0].text == "abcdefghij"
    assert chunks[-1].text == "qrstuvwxyz"


def test_char_strategy_exact_fit():
    txt = "1234567890"  # exactly 10
    chunker = TextChunker(strategy="char", size=10, overlap=3)
    chunks = chunker.chunk(txt)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == 10


def test_sentence_strategy_packing():
    txt = "A. B? C! 这是中文。还有一段？"
    # size 小一点，确保分多段
    chunker = TextChunker(strategy="sentence", size=8, overlap=2)
    chunks = chunker.chunk(txt)
    assert len(chunks) >= 2
    # 每个 chunk 长度不超过 size
    assert all(len(c.text) <= 8 for c in chunks)


def test_sentence_overlap_effect():
    txt = "S1. S2. S3. S4. S5."
    c1 = TextChunker(strategy="sentence", size=6, overlap=0).chunk(txt)
    c2 = TextChunker(strategy="sentence", size=6, overlap=3).chunk(txt)
    # overlap>0 时 chunk 数不减少
    assert len(c2) >= len(c1)

    # 新断言：每个 chunk 的 start 必须严格递增，防止停在同一位置
    starts = [c.start for c in c2]
    assert all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))


def test_invalid_params():
    with pytest.raises(ValueError):
        TextChunker(strategy="char", size=0, overlap=0)
    with pytest.raises(ValueError):
        TextChunker(strategy="char", size=10, overlap=10)
    with pytest.raises(ValueError):
        TextChunker(strategy="invalid", size=10, overlap=1)


def test_empty_text():
    chunker = TextChunker(strategy="char", size=10, overlap=1)
    chunks = chunker.chunk("")
    assert chunks == []
