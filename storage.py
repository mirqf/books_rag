from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from text_utils import estimate_reading_minutes, tokenize, word_count


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "windows-1251", "koi8-r"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def sanitize_filename(name: str) -> str:
    base = Path(name).name or "book.txt"
    cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё0-9._ -]+", "_", base).strip()
    if not cleaned.lower().endswith(".txt"):
        cleaned += ".txt"
    return cleaned or "book.txt"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def looks_like_author(line: str) -> bool:
    words = [part for part in re.split(r"\s+", line.strip()) if part]
    if not 2 <= len(words) <= 5:
        return False
    if len(line) > 80:
        return False
    uppercase_like = 0
    for word in words:
        letter = word[0]
        if letter.isupper() or "." in word:
            uppercase_like += 1
    return uppercase_like >= max(1, len(words) - 1)


def extract_title_author(text: str, fallback_title: str) -> tuple[str, str]:
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip()
        if line:
            lines.append(line)
        if len(lines) >= 20:
            break

    title = fallback_title
    author = "Неизвестен автор"
    if not lines:
        return title, author

    if len(lines) >= 2 and looks_like_author(lines[0]) and len(lines[1]) <= 120:
        author = lines[0]
        title = lines[1]
        return title, author

    if len(lines[0]) <= 120:
        title = lines[0]
    return title, author


def is_heading(paragraph: str) -> bool:
    line = paragraph.strip().strip(":-")
    if not line:
        return False
    if len(line) > 80:
        return False
    if re.fullmatch(r"[IVXLCDM]+", line):
        return True
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    if uppercase_ratio > 0.85 and len(line.split()) <= 8:
        return True
    return line.istitle() and len(line.split()) <= 6


def split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n+", normalized):
        paragraph = " ".join(line.strip() for line in block.splitlines()).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def chunk_text(text: str, target_words: int = 220, overlap_words: int = 55) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[tuple[str, int]] = []
    buffer_words = 0
    chunk_counter = 0

    def emit() -> None:
        nonlocal buffer, buffer_words, chunk_counter
        if not buffer:
            return
        chunk_counter += 1
        chunk_body = "\n\n".join(paragraph for paragraph, _ in buffer).strip()
        location = current_heading or f"Фрагмент {chunk_counter}"
        if current_heading:
            location = f"{current_heading} • фрагмент {chunk_counter}"
        chunks.append((location, chunk_body))

        retained: list[tuple[str, int]] = []
        retained_words = 0
        for paragraph, count in reversed(buffer):
            retained.insert(0, (paragraph, count))
            retained_words += count
            if retained_words >= overlap_words:
                break
        buffer = retained
        buffer_words = retained_words

    for paragraph in split_paragraphs(text):
        if is_heading(paragraph):
            emit()
            current_heading = paragraph
            buffer = []
            buffer_words = 0
            continue

        paragraph_words = len(tokenize(paragraph))
        if paragraph_words == 0:
            continue
        buffer.append((paragraph, paragraph_words))
        buffer_words += paragraph_words
        if buffer_words >= target_words:
            emit()

    if buffer:
        chunk_counter += 1
        chunk_body = "\n\n".join(paragraph for paragraph, _ in buffer).strip()
        location = current_heading or f"Фрагмент {chunk_counter}"
        if current_heading:
            location = f"{current_heading} • фрагмент {chunk_counter}"
        chunks.append((location, chunk_body))

    return chunks


@dataclass
class BookRecord:
    id: int
    file_name: str
    file_path: str
    title: str
    author: str
    added_at: str
    updated_at: str
    size_bytes: int
    word_count: int
    reading_minutes: int
    content_hash: str


@dataclass
class ChunkRecord:
    id: int
    book_id: int
    chunk_index: int
    location: str
    text: str


class LibraryRepository:
    def __init__(self, db_path: Path, library_dir: Path) -> None:
        self.db_path = db_path
        self.library_dir = library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    reading_minutes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    location TEXT NOT NULL,
                    text TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_book_chunk ON chunks(book_id, chunk_index);
                """
            )

    def sync_library(self) -> None:
        files = {path.name: path for path in self.library_dir.glob("*.txt")}
        with self._connect() as connection:
            rows = connection.execute("SELECT id, file_name FROM books").fetchall()
            indexed_names = {row["file_name"] for row in rows}
            missing = indexed_names - set(files)
            for name in missing:
                connection.execute("DELETE FROM books WHERE file_name = ?", (name,))
            connection.commit()

        for path in files.values():
            self._ingest_path(path)

    def _ingest_path(self, path: Path) -> None:
        text = decode_text(path.read_bytes())
        digest = content_hash(text)
        stats_words = word_count(text)
        title, author = extract_title_author(text, path.stem.replace("_", " ").replace("-", " ").strip().title() or path.stem)
        chunks = chunk_text(text)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, added_at, content_hash FROM books WHERE file_name = ?",
                (path.name,),
            ).fetchone()
            if existing and existing["content_hash"] == digest:
                connection.execute(
                    """
                    UPDATE books
                    SET file_path = ?, title = ?, author = ?, size_bytes = ?, word_count = ?, reading_minutes = ?
                    WHERE id = ?
                    """,
                    (
                        str(path),
                        title,
                        author,
                        path.stat().st_size,
                        stats_words,
                        estimate_reading_minutes(stats_words),
                        existing["id"],
                    ),
                )
                connection.commit()
                return

            added_at = existing["added_at"] if existing else utc_now()
            if existing:
                book_id = existing["id"]
                connection.execute(
                    """
                    UPDATE books
                    SET file_path = ?, title = ?, author = ?, updated_at = ?, size_bytes = ?, word_count = ?,
                        reading_minutes = ?, content_hash = ?
                    WHERE id = ?
                    """,
                    (
                        str(path),
                        title,
                        author,
                        utc_now(),
                        path.stat().st_size,
                        stats_words,
                        estimate_reading_minutes(stats_words),
                        digest,
                        book_id,
                    ),
                )
                connection.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO books (
                        file_name, file_path, title, author, added_at, updated_at,
                        size_bytes, word_count, reading_minutes, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        path.name,
                        str(path),
                        title,
                        author,
                        added_at,
                        utc_now(),
                        path.stat().st_size,
                        stats_words,
                        estimate_reading_minutes(stats_words),
                        digest,
                    ),
                )
                book_id = int(cursor.lastrowid)

            connection.executemany(
                """
                INSERT INTO chunks (book_id, chunk_index, location, text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (book_id, index, location, body)
                    for index, (location, body) in enumerate(chunks)
                ),
            )
            connection.commit()

    def save_uploaded_book(self, original_name: str, content: bytes) -> BookRecord:
        decoded = decode_text(content)
        clean_name = sanitize_filename(original_name)
        path = self._unique_path(clean_name)
        path.write_text(decoded, encoding="utf-8")
        self._ingest_path(path)
        return self.get_book_by_filename(path.name)

    def _unique_path(self, file_name: str) -> Path:
        candidate = self.library_dir / file_name
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 2
        while True:
            test = self.library_dir / f"{stem}-{counter}{suffix}"
            if not test.exists():
                return test
            counter += 1

    def delete_book(self, book_id: int) -> Optional[BookRecord]:
        book = self.get_book(book_id)
        if not book:
            return None
        path = Path(book.file_path)
        if path.exists():
            path.unlink()
        with self._connect() as connection:
            connection.execute("DELETE FROM books WHERE id = ?", (book_id,))
            connection.commit()
        return book

    def list_books(self) -> list[BookRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, file_name, file_path, title, author, added_at, updated_at,
                       size_bytes, word_count, reading_minutes, content_hash
                FROM books
                ORDER BY added_at DESC, id DESC
                """
            ).fetchall()
        return [self._row_to_book(row) for row in rows]

    def get_book(self, book_id: int) -> Optional[BookRecord]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, file_name, file_path, title, author, added_at, updated_at,
                       size_bytes, word_count, reading_minutes, content_hash
                FROM books
                WHERE id = ?
                """,
                (book_id,),
            ).fetchone()
        return self._row_to_book(row) if row else None

    def get_book_by_filename(self, file_name: str) -> BookRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, file_name, file_path, title, author, added_at, updated_at,
                       size_bytes, word_count, reading_minutes, content_hash
                FROM books
                WHERE file_name = ?
                """,
                (file_name,),
            ).fetchone()
        if not row:
            raise ValueError(f"Book {file_name} not found")
        return self._row_to_book(row)

    def list_chunks(self) -> list[ChunkRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, book_id, chunk_index, location, text
                FROM chunks
                ORDER BY book_id, chunk_index
                """
            ).fetchall()
        return [ChunkRecord(**dict(row)) for row in rows]

    def _row_to_book(self, row: sqlite3.Row) -> BookRecord:
        return BookRecord(
            id=row["id"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            title=row["title"],
            author=row["author"],
            added_at=row["added_at"],
            updated_at=row["updated_at"],
            size_bytes=row["size_bytes"],
            word_count=row["word_count"],
            reading_minutes=row["reading_minutes"],
            content_hash=row["content_hash"],
        )
