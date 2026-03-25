from __future__ import annotations

import html
import io
from datetime import datetime
from pathlib import Path

from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

from llm_client import OpenRouterClient
from rag_engine import AnswerResponse, RAGService, SearchResponse
from storage import BookRecord, LibraryRepository
from text_utils import format_minutes

router = Router()

repository = LibraryRepository(
    db_path=Path("data") / "catalog.sqlite3",
    library_dir=Path("library"),
)
rag_service = RAGService(repository=repository, llm_client=OpenRouterClient.from_env())


class UserState(StatesGroup):
    waiting_for_upload = State()
    waiting_for_search = State()
    waiting_for_question = State()


def main_keyboard() -> types.ReplyKeyboardMarkup:
    return types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Добавить книгу"), types.KeyboardButton(text="Библиотека")],
            [types.KeyboardButton(text="Поиск фрагментов"), types.KeyboardButton(text="Задать вопрос")],
            [types.KeyboardButton(text="Помощь")],
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие",
    )


def library_keyboard(books: list[BookRecord]) -> types.InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for book in books:
        builder.row(
            InlineKeyboardButton(
                text=f"{book.title} ({book.author})",
                callback_data=f"book:{book.id}",
            )
        )
    builder.row(InlineKeyboardButton(text="Обновить индекс", callback_data="library:refresh"))
    return builder.as_markup()


def book_actions_keyboard(book_id: int) -> types.InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="Удалить книгу", callback_data=f"book:delete:{book_id}"))
    builder.row(InlineKeyboardButton(text="Назад к библиотеке", callback_data="library:list"))
    return builder.as_markup()


def delete_confirmation_keyboard(book_id: int) -> types.InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="Да, удалить", callback_data=f"book:confirm_delete:{book_id}"),
        InlineKeyboardButton(text="Отмена", callback_data=f"book:{book_id}"),
    )
    return builder.as_markup()


def format_datetime(iso_value: str) -> str:
    cleaned = iso_value.replace("Z", "")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return iso_value
    return parsed.strftime("%Y-%m-%d %H:%M")


def format_size(size_bytes: int) -> str:
    units = ["Б", "КБ", "МБ", "ГБ"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size_bytes} Б"


def book_card(book: BookRecord) -> str:
    return (
        f"<b>{html.escape(book.title)}</b>\n"
        f"Автор: {html.escape(book.author)}\n"
        f"Файл: <code>{html.escape(book.file_name)}</code>\n"
        f"Добавлена: {format_datetime(book.added_at)}\n"
        f"Обновлена: {format_datetime(book.updated_at)}\n"
        f"Размер: {format_size(book.size_bytes)}\n"
        f"Слов: {book.word_count:,}\n"
        f"Оценка чтения: около {format_minutes(book.reading_minutes)}"
    )


def search_message(response: SearchResponse) -> str:
    if response.honest_failure:
        return "Ничего надежного не нашел в загруженных книгах. Попробуйте уточнить запрос."

    parts = [f"<b>Лучшие фрагменты по запросу:</b> {html.escape(response.query)}"]
    for index, hit in enumerate(response.hits, start=1):
        parts.append(
            f"\n<b>{index}.</b> {html.escape(hit.book_title)}\n"
            f"Автор: {html.escape(hit.author)}\n"
            f"Источник: {html.escape(hit.location)}\n"
            f"Релевантность: {hit.score:.2f}\n"
            f"{html.escape(hit.text)}"
        )
    return "\n".join(parts)


def answer_message(response: AnswerResponse) -> str:
    return f"<b>Ответ</b> (LLM + Advanced RAG)\n{html.escape(response.answer)}"


def citations_message(response: AnswerResponse) -> str:
    if not response.citations:
        return "Цитат нет: система не нашла надежной опоры в загруженных книгах."
    parts = ["<b>Цитаты-основания</b>"]
    for index, hit in enumerate(response.citations, start=1):
        parts.append(
            f"\n<b>[{index}]</b> {html.escape(hit.book_title)}\n"
            f"{html.escape(hit.location)}\n"
            f"{html.escape(hit.text)}"
        )
    return "\n".join(parts)


async def send_long_message(message: types.Message, text: str) -> None:
    chunk_size = 3800
    if len(text) <= chunk_size:
        await message.answer(text, parse_mode="HTML")
        return

    sections = text.split("\n\n")
    current = []
    current_length = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue
        section_length = len(section) + 2
        if current and current_length + section_length > chunk_size:
            await message.answer("\n\n".join(current), parse_mode="HTML")
            current = [section]
            current_length = len(section)
        else:
            current.append(section)
            current_length += section_length

    if current:
        await message.answer("\n\n".join(current), parse_mode="HTML")


@router.message(Command("start"))
async def start_handler(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    books = await rag_service.list_books()
    await message.answer(
        "RAG-бот для поиска по книгам готов.\n"
        f"Сейчас в библиотеке: {len(books)} книг.\n\n"
        "Что умеет бот:\n"
        "1. Загружать новые .txt-книги.\n"
        "2. Показывать библиотеку и удалять книги.\n"
        "3. Искать фрагменты по запросу.\n"
        "4. Отвечать на вопросы по книгам с цитатами.",
        reply_markup=main_keyboard(),
    )


@router.message(Command("help"))
@router.message(F.text.lower() == "помощь")
async def help_handler(message: types.Message) -> None:
    await message.answer(
        "Команды:\n"
        "/start - главное меню\n"
        "/help - помощь\n\n"
        "Кнопки:\n"
        "• Добавить книгу - загрузить .txt\n"
        "• Библиотека - список книг и удаление\n"
        "• Поиск фрагментов - найти места в книгах\n"
        "• Задать вопрос - получить ответ по книгам\n\n"
        "Сервис требует рабочую конфигурацию OpenRouter: ответы всегда проходят через LLM поверх retrieval.",
        reply_markup=main_keyboard(),
    )


@router.message(F.text.lower() == "добавить книгу")
async def prompt_upload(message: types.Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_for_upload)
    await message.answer("Отправьте `.txt`-файл книги документом.", parse_mode="Markdown")


@router.message(F.text.lower() == "поиск фрагментов")
@router.message(Command("search"))
async def prompt_search(message: types.Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_for_search)
    await message.answer("Напишите запрос в стиле: `Найди, где говорится про...`", parse_mode="Markdown")


@router.message(F.text.lower() == "задать вопрос")
@router.message(Command("ask"))
async def prompt_question(message: types.Message, state: FSMContext) -> None:
    await state.set_state(UserState.waiting_for_question)
    await message.answer("Напишите вопрос по содержанию книг.")


@router.message(F.text.lower() == "библиотека")
@router.message(Command("library"))
async def library_handler(message: types.Message) -> None:
    books = await rag_service.list_books()
    if not books:
        await message.answer("Библиотека пока пустая. Загрузите `.txt`-книгу.", reply_markup=main_keyboard())
        return
    await message.answer(
        f"В библиотеке {len(books)} книг.",
        reply_markup=main_keyboard(),
    )
    await message.answer("Выберите книгу:", reply_markup=library_keyboard(books))


@router.message(UserState.waiting_for_upload, F.document)
@router.message(F.document)
async def upload_document(message: types.Message, state: FSMContext) -> None:
    document = message.document
    if not document or not document.file_name or not document.file_name.lower().endswith(".txt"):
        if await state.get_state() == UserState.waiting_for_upload.state:
            await message.answer("Нужен именно `.txt`-файл.")
        return

    await message.answer("Загружаю книгу и перестраиваю индекс. Это может занять до минуты на больших файлах.")
    buffer = io.BytesIO()
    await message.bot.download(document, destination=buffer)
    book = await rag_service.add_book(document.file_name, buffer.getvalue())
    await state.clear()
    await message.answer(
        "Книга загружена.\n\n" + book_card(book),
        parse_mode="HTML",
        reply_markup=main_keyboard(),
    )


@router.message(UserState.waiting_for_upload)
async def upload_waiting_fallback(message: types.Message) -> None:
    await message.answer("Сейчас жду `.txt`-файл книги документом.")


@router.message(UserState.waiting_for_search, F.text)
async def run_search(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Ищу лучшие фрагменты...")
    response = await rag_service.search(message.text)
    await send_long_message(message, search_message(response))


@router.message(UserState.waiting_for_question, F.text)
async def run_question(message: types.Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Собираю контекст и формирую ответ...")
    try:
        response = await rag_service.answer_question(message.text)
    except RuntimeError as exc:
        await message.answer(
            f"Не удалось получить ответ от LLM: {html.escape(str(exc))}",
            parse_mode="HTML",
        )
        return
    await send_long_message(message, answer_message(response))
    await send_long_message(message, citations_message(response))


@router.callback_query(F.data == "library:list")
async def callback_library_list(callback: CallbackQuery) -> None:
    books = await rag_service.list_books()
    if not books:
        await callback.message.edit_text("Библиотека пуста.")
    else:
        await callback.message.edit_text("Выберите книгу:", reply_markup=library_keyboard(books))
    await callback.answer()


@router.callback_query(F.data == "library:refresh")
async def callback_library_refresh(callback: CallbackQuery) -> None:
    await callback.message.edit_text("Обновляю библиотеку и индекс...")
    await rag_service.initialize()
    books = await rag_service.list_books()
    if not books:
        await callback.message.edit_text("Библиотека пуста.")
    else:
        await callback.message.edit_text("Выберите книгу:", reply_markup=library_keyboard(books))
    await callback.answer("Индекс обновлен")


@router.callback_query(F.data.regexp(r"^book:\d+$"))
async def callback_book_details(callback: CallbackQuery) -> None:
    if not callback.data:
        await callback.answer()
        return
    _, raw_id = callback.data.split(":", maxsplit=1)
    if not raw_id.isdigit():
        await callback.answer()
        return
    book = await rag_service.get_book(int(raw_id))
    if not book:
        await callback.message.edit_text("Книга не найдена.")
        await callback.answer()
        return
    await callback.message.edit_text(
        book_card(book),
        parse_mode="HTML",
        reply_markup=book_actions_keyboard(book.id),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("book:delete:"))
async def callback_book_delete(callback: CallbackQuery) -> None:
    if not callback.data:
        await callback.answer()
        return
    book_id = int(callback.data.rsplit(":", maxsplit=1)[-1])
    book = await rag_service.get_book(book_id)
    if not book:
        await callback.message.edit_text("Книга уже удалена.")
        await callback.answer()
        return
    await callback.message.edit_text(
        f"Удалить книгу <b>{html.escape(book.title)}</b>?",
        parse_mode="HTML",
        reply_markup=delete_confirmation_keyboard(book_id),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("book:confirm_delete:"))
async def callback_book_confirm_delete(callback: CallbackQuery) -> None:
    if not callback.data:
        await callback.answer()
        return
    book_id = int(callback.data.rsplit(":", maxsplit=1)[-1])
    book = await rag_service.delete_book(book_id)
    if not book:
        await callback.message.edit_text("Книга уже удалена.")
        await callback.answer()
        return
    books = await rag_service.list_books()
    if not books:
        await callback.message.edit_text(f"Книга «{html.escape(book.title)}» удалена. Библиотека теперь пуста.", parse_mode="HTML")
    else:
        await callback.message.edit_text(
            f"Книга «{html.escape(book.title)}» удалена.\n\nВыберите книгу:",
            parse_mode="HTML",
            reply_markup=library_keyboard(books),
        )
    await callback.answer("Книга удалена")


@router.message(F.text)
async def fallback_text_router(message: types.Message) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    await message.answer(
        "Используйте кнопки меню: загрузка книги, библиотека, поиск фрагментов или вопрос по книгам.",
        reply_markup=main_keyboard(),
    )
