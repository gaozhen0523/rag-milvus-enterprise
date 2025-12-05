# libs/chunking/text_chunker.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start: int  # char-level start offset in original text
    end: int  # char-level end offset (exclusive)
    meta: dict | None = None


class TextChunker:
    """
    通用文本切分器：
      - strategy = 'char' 固定字符窗口 + overlap
      - strategy = 'sentence' 句子粒度打包到 size 限制内 + overlap（按字符计算）
    参数：
      size: 每个chunk的最大字符数（> 0）
      overlap: 相邻chunk的重叠字符数（0 <= overlap < size）
    """

    SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!\?。！？；;])\s*|\n+")

    def __init__(
        self,
        strategy: Literal["char", "sentence"] = "char",
        size: int = 800,
        overlap: int = 100,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        if overlap < 0 or overlap >= size:
            raise ValueError("0 <= overlap < size must hold")
        if strategy not in ("char", "sentence"):
            raise ValueError("strategy must be 'char' or 'sentence'")

        self.strategy = strategy
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, *, meta: dict | None = None) -> list[Chunk]:
        text = text or ""
        if not text:
            return []

        if self.strategy == "char":
            return self._chunk_by_char(text, meta=meta)
        else:
            return self._chunk_by_sentence(text, meta=meta)

    # ---------- strategy: char ----------
    def _chunk_by_char(self, text: str, *, meta: dict | None) -> list[Chunk]:
        chunks: list[Chunk] = []

        pos = 0
        cid = 0
        n = len(text)
        while pos < n:
            end = min(pos + self.size, n)
            chunks.append(
                Chunk(chunk_id=cid, text=text[pos:end], start=pos, end=end, meta=meta)
            )
            cid += 1
            if end == n:
                break
            pos = end - self.overlap
        return chunks

    # ---------- strategy: sentence ----------
    def _split_sentences_with_spans(self, text: str) -> list[tuple[int, int, str]]:
        """
        以标点和换行作为边界，保留 (start, end, sentence_text)。
        句子最少包含非空字符；连续空白会被合并到边界。
        """
        spans: list[tuple[int, int, str]] = []
        last = 0
        for m in self.SENTENCE_SPLIT_RE.finditer(text):
            end = m.start()  # boundary BEFORE delimiter match
            if end > last:
                seg = text[last:end].strip()
                if seg:
                    # 重新定位去掉两侧空白后的精确span
                    seg_start = text.find(seg, last, end)
                    seg_end = seg_start + len(seg)
                    spans.append((seg_start, seg_end, seg))
            last = m.end()
        # tail
        if last < len(text):
            seg = text[last:].strip()
            if seg:
                seg_start = text.find(seg, last)
                seg_end = seg_start + len(seg)
                spans.append((seg_start, seg_end, seg))
        # 若全为空或无分句，退化成整段
        if not spans:
            spans = [(0, len(text), text)]
        return spans

    def _chunk_by_sentence(self, text: str, *, meta: dict | None) -> list[Chunk]:
        sentences = self._split_sentences_with_spans(text)
        chunks: list[Chunk] = []
        n = len(sentences)
        i = 0
        cid = 0

        while i < n:
            # 从第 i 个句子开始尽可能打包，使 chunk 文本长度 <= size
            start_span = sentences[i][0]
            end_span = sentences[i][1]
            j = i
            while j + 1 < n:
                next_text_len = sentences[j + 1][1] - start_span
                if next_text_len <= self.size:
                    j += 1
                    end_span = sentences[j][1]
                else:
                    break

            chunk_text = text[start_span:end_span]
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=chunk_text,
                    start=start_span,
                    end=end_span,
                    meta=meta,
                )
            )
            cid += 1

            if j == n - 1:
                break

            # 计算带 overlap 的下一窗口起点（按字符），再映射到 sentence 边界
            desired_next_start = max(end_span - self.overlap, start_span)
            # 找到第一个 sentence whose start >= desired_next_start
            next_i = j + 1
            for k in range(i, j + 1):
                if sentences[k][0] >= desired_next_start:
                    next_i = k
                    break
            # ✅ 关键：防止 next_i 回到 i 导致死循环
            if next_i <= i:
                next_i = i + 1

            i = next_i

        return chunks
