from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "а",
    "без",
    "более",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всего",
    "вы",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "её",
    "если",
    "есть",
    "еще",
    "ещё",
    "же",
    "за",
    "здесь",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "когда",
    "кто",
    "ли",
    "либо",
    "между",
    "меня",
    "мне",
    "может",
    "можно",
    "мой",
    "мы",
    "на",
    "над",
    "надо",
    "наш",
    "не",
    "него",
    "нее",
    "неё",
    "нет",
    "ни",
    "них",
    "но",
    "ну",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "по",
    "под",
    "после",
    "потом",
    "потому",
    "почти",
    "при",
    "про",
    "раз",
    "с",
    "сам",
    "сво",
    "себе",
    "себя",
    "со",
    "так",
    "также",
    "такой",
    "там",
    "те",
    "тем",
    "то",
    "того",
    "тоже",
    "только",
    "том",
    "ты",
    "у",
    "уж",
    "уже",
    "хоть",
    "чего",
    "чей",
    "чем",
    "через",
    "что",
    "чтоб",
    "чтобы",
    "эта",
    "эти",
    "это",
    "я",
}

STEM_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ией",
    "ией",
    "ией",
    "ого",
    "ему",
    "ому",
    "ее",
    "ие",
    "ые",
    "ое",
    "ей",
    "ий",
    "ый",
    "ой",
    "ем",
    "им",
    "ым",
    "ом",
    "их",
    "ых",
    "ую",
    "юю",
    "ая",
    "яя",
    "ою",
    "ею",
    "ия",
    "ья",
    "ьям",
    "иях",
    "ах",
    "ях",
    "иям",
    "ов",
    "ев",
    "ом",
    "ем",
    "ам",
    "ям",
    "ию",
    "ью",
    "ия",
    "ья",
    "а",
    "е",
    "и",
    "й",
    "о",
    "у",
    "ы",
    "ь",
    "ю",
    "я",
)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = text.replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def word_count(text: str) -> int:
    return len(tokenize(text))


def stem_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    if len(token) <= 3:
        return token
    for suffix in STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in tokenize(text):
        if token in STOPWORDS or token.isdigit():
            continue
        stem = stem_token(token)
        if len(stem) < 3:
            continue
        terms.append(stem)
    return terms


def term_counter(text: str) -> Counter[str]:
    return Counter(content_terms(text))


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text.strip() else []


def char_ngrams(text: str, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", normalize_text(text)).replace(" ", "")
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def estimate_reading_minutes(words: int, words_per_minute: int = 180) -> int:
    if words <= 0:
        return 1
    return max(1, math.ceil(words / words_per_minute))


def format_minutes(minutes: int) -> str:
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours} ч {mins} мин"
    if hours:
        return f"{hours} ч"
    return f"{mins} мин"
