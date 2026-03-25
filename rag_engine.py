from __future__ import annotations

import asyncio
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from llm_client import LLMContext, OpenRouterClient
from storage import BookRecord, ChunkRecord, LibraryRepository
from text_utils import char_ngrams, content_terms, normalize_text, split_sentences, term_counter


@dataclass
class SearchHit:
    chunk_id: int
    chunk_index: int
    book_id: int
    book_title: str
    author: str
    location: str
    text: str
    full_text: str
    score: float


@dataclass
class SearchResponse:
    query: str
    hits: list[SearchHit]
    honest_failure: bool


@dataclass
class AnswerResponse:
    question: str
    answer: str
    citations: list[SearchHit]
    honest_failure: bool
    used_llm: bool


@dataclass
class IndexedChunk:
    chunk_id: int
    book_id: int
    chunk_index: int
    book_title: str
    author: str
    location: str
    text: str
    normalized_text: str
    term_list: list[str]
    terms: Counter[str]
    unique_terms: frozenset[str]
    length: int


def cosine_overlap(query_terms: Iterable[str], chunk_terms: Iterable[str]) -> float:
    query_counter = Counter(query_terms)
    chunk_counter = Counter(chunk_terms)
    if not query_counter or not chunk_counter:
        return 0.0
    dot = sum(query_counter[term] * chunk_counter.get(term, 0) for term in query_counter)
    if dot == 0:
        return 0.0
    query_norm = math.sqrt(sum(value * value for value in query_counter.values()))
    chunk_norm = math.sqrt(sum(value * value for value in chunk_counter.values()))
    if query_norm == 0 or chunk_norm == 0:
        return 0.0
    return dot / (query_norm * chunk_norm)


class RAGService:
    def __init__(self, repository: LibraryRepository, llm_client: Optional[OpenRouterClient] = None) -> None:
        self.repository = repository
        self.llm_client = llm_client
        self.books: dict[int, BookRecord] = {}
        self.indexed_chunks: list[IndexedChunk] = []
        self.idf: dict[str, float] = {}
        self.avg_chunk_length = 1.0

    async def initialize(self) -> None:
        await asyncio.to_thread(self.reload)

    async def validate_llm(self) -> None:
        if not self.llm_client:
            raise RuntimeError("LLM configuration is missing. Set OPENROUTER_API_KEY and optionally OPENROUTER_MODEL.")
        await asyncio.to_thread(self.llm_client.validate_configuration)

    def reload(self) -> None:
        self.repository.sync_library()
        books = self.repository.list_books()
        chunks = self.repository.list_chunks()
        self.books = {book.id: book for book in books}
        self.indexed_chunks = []

        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            book = self.books.get(chunk.book_id)
            if not book:
                continue
            terms = term_counter(chunk.text)
            unique_terms = frozenset(terms)
            indexed = IndexedChunk(
                chunk_id=chunk.id,
                book_id=chunk.book_id,
                chunk_index=chunk.chunk_index,
                book_title=book.title,
                author=book.author,
                location=chunk.location,
                text=chunk.text,
                normalized_text=normalize_text(chunk.text),
                term_list=content_terms(chunk.text),
                terms=terms,
                unique_terms=unique_terms,
                length=max(1, sum(terms.values())),
            )
            self.indexed_chunks.append(indexed)
            document_frequency.update(unique_terms)

        total_chunks = max(1, len(self.indexed_chunks))
        self.avg_chunk_length = (
            sum(chunk.length for chunk in self.indexed_chunks) / len(self.indexed_chunks)
            if self.indexed_chunks
            else 1.0
        )
        self.idf = {
            term: math.log(1 + (total_chunks - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    async def add_book(self, file_name: str, content: bytes) -> BookRecord:
        book = await asyncio.to_thread(self.repository.save_uploaded_book, file_name, content)
        await asyncio.to_thread(self.reload)
        return book

    async def delete_book(self, book_id: int) -> Optional[BookRecord]:
        book = await asyncio.to_thread(self.repository.delete_book, book_id)
        await asyncio.to_thread(self.reload)
        return book

    async def list_books(self) -> list[BookRecord]:
        return await asyncio.to_thread(self.repository.list_books)

    async def get_book(self, book_id: int) -> Optional[BookRecord]:
        return await asyncio.to_thread(self.repository.get_book, book_id)

    async def search(self, query: str, limit: int = 5) -> SearchResponse:
        return await asyncio.to_thread(self._search_sync, query, limit)

    async def answer_question(self, question: str, limit: int = 5) -> AnswerResponse:
        return await asyncio.to_thread(self._answer_sync, question, limit)

    def _search_sync(self, query: str, limit: int = 5) -> SearchResponse:
        hits = self._retrieve(query, limit=limit, mode="search")
        return SearchResponse(query=query, hits=hits, honest_failure=not hits)

    def _answer_sync(self, question: str, limit: int = 5) -> AnswerResponse:
        if not self.llm_client or not self.llm_client.enabled:
            raise RuntimeError("LLM is required for answers, but OpenRouter is not configured.")

        hits = self._fused_answer_hits(question, limit=max(limit + 3, 8))
        response = self._llm_answer_with_retry(question, hits, limit=limit)
        return response

    def _answer_terms(self, query: str) -> list[str]:
        terms = content_terms(query)
        if terms:
            return terms
        return [term for term in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", query.lower()) if term]

    def _heuristic_query_variants(self, question: str) -> list[str]:
        variants = [question.strip()]
        clean = re.sub(r"[?!.]+$", "", question.strip())
        variants.append(clean)

        who_match = re.search(r"кто так[ао]й\s+(.+)", clean, flags=re.I)
        if who_match:
            entity = who_match.group(1).strip(" \"«»")
            variants.extend(
                [
                    f"кто такой {entity}",
                    f"описание {entity}",
                    f"кем был {entity}",
                    f"роль {entity} в романе",
                    f"{entity} был",
                ]
            )

        if "что произошло" in clean.lower():
            base = re.sub(r"(?i)^что произошло\s*", "", clean).strip(" ,")
            if base:
                variants.extend(
                    [
                        clean,
                        f"события {base}",
                        f"эпизод {base}",
                        f"что случилось {base}",
                    ]
                )

        deduped = []
        seen = set()
        for variant in variants:
            normalized = normalize_text(variant)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(variant)
        return deduped

    def _build_query_variants(self, question: str) -> list[str]:
        return self._heuristic_query_variants(question)

    def _mentioned_titles(self, query: str) -> set[int]:
        normalized_query = normalize_text(query)
        matched: set[int] = set()
        quoted_titles = re.findall(r"[\"«](.*?)[\"»]", query)
        for book in self.books.values():
            title_norm = normalize_text(book.title)
            if any(normalize_text(title) in title_norm or title_norm in normalize_text(title) for title in quoted_titles):
                matched.add(book.id)
                continue
            title_terms = [term for term in content_terms(book.title) if len(term) >= 4]
            overlap = sum(1 for term in set(title_terms) if term in normalized_query)
            if title_terms and overlap >= min(2, len(set(title_terms))):
                matched.add(book.id)
        return matched

    def _bm25(self, query_terms: list[str], chunk: IndexedChunk) -> float:
        score = 0.0
        k1 = 1.6
        b = 0.72
        for term in query_terms:
            frequency = chunk.terms.get(term, 0)
            if not frequency:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = frequency + k1 * (1 - b + b * chunk.length / self.avg_chunk_length)
            score += idf * (frequency * (k1 + 1)) / denominator
        return score

    def _sentence_bonus(self, query: str, chunk: IndexedChunk) -> float:
        query_terms = self._answer_terms(query)
        best_score = 0.0
        for sentence in split_sentences(chunk.text):
            score = cosine_overlap(query_terms, content_terms(sentence))
            if score > best_score:
                best_score = score
        return best_score

    def _proximity_bonus(self, query_terms: list[str], chunk: IndexedChunk) -> float:
        required = set(query_terms)
        if len(required) < 2:
            return 0.0

        positions = [(index, term) for index, term in enumerate(chunk.term_list) if term in required]
        covered_terms = {term for _, term in positions}
        if len(covered_terms) < 2:
            return 0.0

        best_window = None
        counts = Counter()
        left = 0
        covered = 0
        for right, (_, term) in enumerate(positions):
            counts[term] += 1
            if counts[term] == 1:
                covered += 1

            while covered == len(required):
                span = positions[right][0] - positions[left][0] + 1
                if best_window is None or span < best_window:
                    best_window = span
                left_term = positions[left][1]
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    covered -= 1
                left += 1

        if best_window is None:
            return len(covered_terms) / len(required) * 0.2
        return min(1.35, (len(required) * 6.0) / best_window)

    def _definition_bonus(self, query: str, chunk: IndexedChunk) -> float:
        lowered = query.lower()
        if "кто такой" not in lowered and "кто такая" not in lowered:
            return 0.0
        cues = ("был", "была", "это", "князь", "граф", "генерал", "фельдмаршал", "главнокоманд", "дочь", "сын")
        return 0.55 if any(cue in chunk.normalized_text for cue in cues) else 0.0

    def _rank_chunks(self, query: str) -> list[tuple[float, IndexedChunk]]:
        query_terms = self._answer_terms(query)
        if not query_terms or not self.indexed_chunks:
            return []

        query_ngrams = char_ngrams(query)
        mentioned_titles = self._mentioned_titles(query)
        candidates: list[tuple[float, IndexedChunk]] = []

        for chunk in self.indexed_chunks:
            base = self._bm25(query_terms, chunk)
            matched_terms = sum(1 for term in set(query_terms) if term in chunk.unique_terms)
            coverage = matched_terms / max(1, len(set(query_terms)))
            if coverage == 0 and base < 0.1:
                continue

            phrase_bonus = 0.45 if normalize_text(query)[:120] and normalize_text(query)[:120] in chunk.normalized_text else 0.0
            sentence_bonus = self._sentence_bonus(query, chunk)
            lexical_bonus = cosine_overlap(query_terms, chunk.term_list)
            proximity_bonus = self._proximity_bonus(query_terms, chunk)
            definition_bonus = self._definition_bonus(query, chunk)

            ngram_bonus = 0.0
            if query_ngrams:
                chunk_ngrams = char_ngrams(chunk.text[:1400])
                union = len(query_ngrams | chunk_ngrams)
                if union:
                    ngram_bonus = len(query_ngrams & chunk_ngrams) / union

            title_bonus = 0.4 if mentioned_titles and chunk.book_id in mentioned_titles else 0.0
            score = (
                base * 1.7
                + coverage * 2.3
                + sentence_bonus * 1.6
                + lexical_bonus
                + proximity_bonus * 1.7
                + definition_bonus
                + ngram_bonus * 1.1
                + phrase_bonus
                + title_bonus
            )
            if score > 0.55:
                candidates.append((score, chunk))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _select_hits(
        self,
        query: str,
        ranked_candidates: Sequence[tuple[float, IndexedChunk]],
        limit: int,
        min_top_score: float,
    ) -> list[SearchHit]:
        if not ranked_candidates:
            return []

        selected: list[tuple[float, IndexedChunk]] = []
        seen_excerpts: list[str] = []
        for score, chunk in ranked_candidates[:80]:
            if len(selected) >= limit:
                break

            penalty = 0.0
            for _, chosen in selected:
                overlap = len(chunk.unique_terms & chosen.unique_terms) / max(1, len(chunk.unique_terms | chosen.unique_terms))
                if chunk.book_id == chosen.book_id and abs(chunk.chunk_index - chosen.chunk_index) <= 1:
                    overlap = max(overlap, 0.82)
                penalty = max(penalty, overlap)

            mmr_score = score - penalty * 0.35
            if mmr_score <= 0.4 and selected:
                continue

            excerpt = self._best_excerpt(query, chunk.text)
            normalized_excerpt = normalize_text(excerpt)
            if normalized_excerpt in seen_excerpts:
                continue

            selected.append((mmr_score, chunk))
            seen_excerpts.append(normalized_excerpt)

        if not selected:
            return []

        top_score = max(score for score, _ in selected)
        if top_score < min_top_score:
            return []

        return [
            SearchHit(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                book_id=chunk.book_id,
                book_title=chunk.book_title,
                author=chunk.author,
                location=chunk.location,
                text=self._best_excerpt(query, chunk.text),
                full_text=chunk.text,
                score=score,
            )
            for score, chunk in sorted(selected, key=lambda item: item[0], reverse=True)[:limit]
        ]

    def _retrieve(self, query: str, limit: int = 5, mode: str = "search") -> list[SearchHit]:
        ranked = self._rank_chunks(query)
        min_top_score = 0.78 if mode == "search" else 0.92
        return self._select_hits(query, ranked, limit=limit, min_top_score=min_top_score)

    def _fused_answer_hits(self, question: str, limit: int = 8) -> list[SearchHit]:
        variants = self._build_query_variants(question)
        fused_scores: defaultdict[int, float] = defaultdict(float)
        best_scores: dict[int, float] = {}
        chunk_by_id: dict[int, IndexedChunk] = {}

        for query in variants:
            ranked = self._rank_chunks(query)[:14]
            for rank, (score, chunk) in enumerate(ranked, start=1):
                fused_scores[chunk.chunk_id] += 1.0 / (60 + rank)
                best_scores[chunk.chunk_id] = max(best_scores.get(chunk.chunk_id, 0.0), score)
                chunk_by_id[chunk.chunk_id] = chunk

        ranked_fused = sorted(
            (
                (fused_scores[chunk_id] + best_scores.get(chunk_id, 0.0) * 0.02, chunk_by_id[chunk_id])
                for chunk_id in fused_scores
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return self._select_hits(question, ranked_fused, limit=limit, min_top_score=0.01)

    def _sentence_relevance(self, query_terms: list[str], sentence: str) -> float:
        sentence_terms = content_terms(sentence)
        if not sentence_terms:
            return 0.0
        unique_query_terms = set(query_terms)
        coverage = sum(1 for term in unique_query_terms if term in sentence_terms) / max(1, len(unique_query_terms))
        rarity = sum(self.idf.get(term, 0.0) for term in unique_query_terms if term in sentence_terms)
        return coverage * 2.4 + cosine_overlap(query_terms, sentence_terms) * 1.5 + rarity * 0.25

    def _best_excerpt(self, query: str, text: str) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return text[:700]
        query_terms = self._answer_terms(query)
        ranked = sorted(
            sentences,
            key=lambda sentence: self._sentence_relevance(query_terms, sentence),
            reverse=True,
        )
        excerpt = " ".join(ranked[:3]).strip()
        if len(excerpt) > 700:
            excerpt = excerpt[:697].rstrip() + "..."
        return excerpt or text[:700]

    def _contexts_from_hits(self, hits: Sequence[SearchHit]) -> list[LLMContext]:
        return [
            LLMContext(
                citation_id=index + 1,
                book_title=hit.book_title,
                location=hit.location,
                text=hit.full_text[:2600],
            )
            for index, hit in enumerate(hits)
        ]

    def _citations_by_ids(self, hits: Sequence[SearchHit], citation_ids: Sequence[int], limit: int) -> list[SearchHit]:
        selected = []
        for citation_id in citation_ids:
            if isinstance(citation_id, int) and 1 <= citation_id <= len(hits):
                selected.append(hits[citation_id - 1])
        deduped = []
        seen = set()
        for hit in selected:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            deduped.append(hit)
        if deduped:
            return deduped[:limit]
        return list(hits[:limit])

    def _llm_answer_with_retry(self, question: str, hits: list[SearchHit], limit: int) -> AnswerResponse:
        assert self.llm_client is not None

        attempts = 0
        current_hits = hits
        result = None
        while attempts < 2:
            result = self.llm_client.synthesize_answer(question, self._contexts_from_hits(current_hits))
            sufficient = bool(result.get("sufficient"))
            needs_retry = bool(result.get("needs_retry"))
            retry_query = str(result.get("retry_query", "")).strip()

            if sufficient or not needs_retry or not retry_query:
                break
            current_hits = self._fused_answer_hits(retry_query, limit=max(limit + 3, 8))
            attempts += 1

        if result is None:
            raise RuntimeError("LLM did not return an answer")

        answer = str(result.get("answer", "")).strip() or "Недостаточно данных в загруженных книгах."
        sufficient = bool(result.get("sufficient")) and answer != "Недостаточно данных в загруженных книгах."
        raw_citation_ids = result.get("citation_ids", [])
        citation_ids = [value for value in raw_citation_ids if isinstance(value, int)] if isinstance(raw_citation_ids, list) else []
        citations = self._citations_by_ids(current_hits, citation_ids, limit=limit)

        return AnswerResponse(
            question=question,
            answer=answer,
            citations=citations,
            honest_failure=not sufficient,
            used_llm=True,
        )
