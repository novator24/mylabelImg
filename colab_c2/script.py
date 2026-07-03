# Шебанг: запуск скрипта через интерпретатор Python 3
#!/usr/bin/env python3
# Декларация кодировки исходного файла UTF-8
# -*- coding: utf-8 -*-
# Модульная документация: назначение скрипта
"""Сбор равномерной выборки снимков талька через OpenRouter (DeepSeek VL)."""

# Импорт отложенной оценки аннотаций типов
from __future__ import annotations

# Импорт кодирования бинарных данных в base64
import base64
# Импорт работы с JSON
import json
# Импорт определения MIME-типов файлов
import mimetypes
# Импорт доступа к переменным окружения
import os
# Импорт генератора псевдослучайных чисел
import random
# Импорт регулярных выражений
import re
# Импорт операций с файловой системой (копирование, удаление)
import shutil
# Импорт пауз между повторными запросами
import time
# Импорт объектно-ориентированных путей к файлам
from pathlib import Path
# Импорт универсального типа Any для аннотаций
from typing import Any

# Импорт HTTP-клиента для запросов к OpenRouter
import requests

# Корневая директория проекта (родитель каталога colab_c2)
ROOT = Path(__file__).resolve().parent.parent
# Каталог с исходными изображениями all/100, all/200, all/300
ALL_DIR = ROOT / "all"
# Путь к шаблону промпта для классификации
PROMPT_FILE = ROOT / "prompt.txt"
# Путь к файлу с критериями разметки талька
TEST_C2_FILE = ROOT / "test_c2.txt"
# Каталог для сохранения результатов выборки
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# Путь к JSON-манифесту собранной выборки
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
# Подкаталог с копиями изображений, сгруппированных по классам
SAMPLE_DIR = OUTPUT_DIR / "sample"

# URL эндпоинта chat/completions OpenRouter API
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Имя модели DeepSeek VL; переопределяется через OPENROUTER_MODEL
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-vl2-small")

# Допустимые расширения файлов изображений
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# Целевое количество снимков на каждый класс в выборке
TARGET_PER_CLASS = 100
# Максимум попыток классификации (лимит API-вызовов)
MAX_ATTEMPTS = int(os.environ.get("MAX_CLASSIFY_ATTEMPTS", "5000"))
# Таймаут одного HTTP-запроса к OpenRouter в секундах
REQUEST_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", "120"))
# Число повторных попыток при ошибке запроса
RETRY_COUNT = 3
# Базовая задержка между повторами в секундах
RETRY_DELAY_SEC = 2.0

# Справочник из 10 классов талька с id, меткой и описанием
CLASSES: list[dict[str, Any]] = [
    # Класс 0: прочее / не тальк / неопределимо
    {"id": 0, "name": "0_other", "title": "другое / не тальк / не определяется"},
    # Класс 1: чешуйчатый тальк с крупными листоватыми агрегатами
    {"id": 1, "name": "1_talc_flaky", "title": "тальк_чешуйчатый (крупные чешуйки, листоватые агрегаты)"},
    # Класс 2: тонкий мелкочешуйчатый тальк
    {"id": 2, "name": "2_talc_thin", "title": "тальк_тонкий (мелкочешуйчатый, пластинки <1–2 мкм)"},
    # Класс 3: массивный стеатит без выраженной чешуйчатости
    {"id": 3, "name": "3_talc_massive", "title": "тальк_массивный (стеатит, фарфоровидный, без чешуйчатости)"},
    # Класс 4: тальк в ассоциации с серпентином
    {"id": 4, "name": "4_talc_serpentinic", "title": "тальк с серпентином"},
    # Класс 5: тальк в ассоциации с хлоритом
    {"id": 5, "name": "5_talc_chlorite", "title": "тальк с хлоритом"},
    # Класс 6: тальк с магнезитом или доломитом
    {"id": 6, "name": "6_talc_magnesite", "title": "тальк с магнезитом/доломитом"},
    # Класс 7: тальк в ассоциации с кварцем
    {"id": 7, "name": "7_talc_quartz", "title": "тальк с кварцем"},
    # Класс 8: волокнистый асбестоподобный тальк
    {"id": 8, "name": "8_talc_fibrous", "title": "тальк_асбестоподобный (волокнистый)"},
    # Класс 9: гидротермально или метаморфически изменённый тальк
    {"id": 9, "name": "9_talc_altered", "title": "тальк_изменённый (гидротермально/метаморфически)"},
]

# Быстрый поиск класса по числовому идентификатору
CLASS_BY_ID = {c["id"]: c for c in CLASSES}
# Быстрый поиск класса по строковой метке вида 1_talc_flaky
CLASS_BY_NAME = {c["name"]: c for c in CLASSES}
# Соответствие устаревших коротких имён классам из test_c2.txt
LEGACY_NAME_MAP = {
    # Устаревшее имя other → класс 0
    "other": 0,
    # Устаревшее имя talc_flaky → класс 1
    "talc_flaky": 1,
    # Устаревшее имя talc_thin → класс 2
    "talc_thin": 2,
    # Устаревшее имя talc_massive → класс 3
    "talc_massive": 3,
    # Устаревшее имя talc_serpentinic → класс 4
    "talc_serpentinic": 4,
    # Устаревшее имя talc_chlorite → класс 5
    "talc_chlorite": 5,
    # Устаревшее имя talc_magnesite → класс 6
    "talc_magnesite": 6,
    # Устаревшее имя talc_quartz → класс 7
    "talc_quartz": 7,
    # Устаревшее имя talc_fibrous → класс 8
    "talc_fibrous": 8,
    # Устаревшее имя talc_weathered → класс 9
    "talc_weathered": 9,
    # Устаревшее имя talc_altered → класс 9
    "talc_altered": 9,
    # Устаревшее составное имя talc_weathered/altered → класс 9
    "talc_weathered/altered": 9,
}


# Главная функция: оркестрация всех шагов пайплайна
def main() -> None:
    # Шаг 1: загрузка конфигурации и проверка окружения
    def step_1_load_config() -> dict[str, Any]:
        # Чтение API-ключа OpenRouter из переменной окружения
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        # Проверка, что ключ задан
        if not api_key:
            # Исключение при отсутствии ключа
            raise RuntimeError("Установите переменную окружения OPENROUTER_API_KEY")

        # Проверка существования файла шаблона промпта
        if not PROMPT_FILE.is_file():
            # Исключение, если prompt.txt не найден
            raise FileNotFoundError(f"Не найден файл промпта: {PROMPT_FILE}")
        # Проверка существования файла критериев test_c2.txt
        if not TEST_C2_FILE.is_file():
            # Исключение, если test_c2.txt не найден
            raise FileNotFoundError(f"Не найден файл критериев: {TEST_C2_FILE}")
        # Проверка существования каталога all с изображениями
        if not ALL_DIR.is_dir():
            # Исключение, если каталог all отсутствует
            raise FileNotFoundError(f"Не найден каталог изображений: {ALL_DIR}")

        # Формирование словаря конфигурации для последующих шагов
        config = {
            # Секретный ключ API (не выводится в лог)
            "api_key": api_key,
            # Идентификатор модели OpenRouter
            "model": DEFAULT_MODEL,
            # Корень проекта
            "root": ROOT,
            # Путь к каталогу all
            "all_dir": ALL_DIR,
            # Путь к каталогу output
            "output_dir": OUTPUT_DIR,
            # Путь к каталогу sample внутри output
            "sample_dir": SAMPLE_DIR,
            # Путь к файлу manifest.json
            "manifest_file": MANIFEST_FILE,
            # Целевое число снимков на класс
            "target_per_class": TARGET_PER_CLASS,
            # Seed для воспроизводимой случайной перестановки
            "seed": int(os.environ.get("RANDOM_SEED", "42")),
        }
        # Печать конфигурации шага 1 без api_key
        print(json.dumps({"step": 1, "config": {k: str(v) for k, v in config.items() if k != "api_key"}}, ensure_ascii=False, indent=2))
        # Возврат готовой конфигурации
        return config

    # Шаг 2: сбор и перемешивание путей ко всем изображениям
    def step_2_collect_images(config: dict[str, Any]) -> list[Path]:
        # Список корневых подкаталогов 100, 200, 300 внутри all
        roots = [config["all_dir"] / name for name in ("100", "200", "300")]
        # Аккумулятор путей к найденным изображениям
        images: list[Path] = []
        # Обход каждого корневого подкаталога
        for root in roots:
            # Пропуск отсутствующих каталогов
            if not root.is_dir():
                # Переход к следующему корню
                continue
            # Рекурсивный обход всех файлов в поддереве
            for path in root.rglob("*"):
                # Отбор только файлов с допустимым расширением изображения
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    # Добавление абсолютного пути в список
                    images.append(path.resolve())

        # Удаление дубликатов и сортировка для стабильности
        images = sorted(set(images))
        # Инициализация генератора случайных чисел фиксированным seed
        random.seed(config["seed"])
        # Случайная перестановка порядка обработки снимков
        random.shuffle(images)

        # Сводка результата шага 2 для печати
        result = {"step": 2, "total_images": len(images), "roots": [str(r) for r in roots], "preview": [str(p) for p in images[:5]]}
        # Вывод JSON-отчёта шага 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Возврат списка путей к изображениям
        return images

    # Шаг 3: сборка итогового промпта из шаблона и критериев
    def step_3_build_prompt(config: dict[str, Any]) -> str:
        # Чтение текста шаблона prompt.txt
        template = PROMPT_FILE.read_text(encoding="utf-8")
        # Чтение полного текста критериев из test_c2.txt
        criteria_source = TEST_C2_FILE.read_text(encoding="utf-8")

        # Буфер строк критериев, извлечённых из test_c2.txt
        criteria_lines: list[str] = []
        # Флаг начала захвата блока критериев
        capture = False
        # Построчный разбор test_c2.txt
        for line in criteria_source.splitlines():
            # Обрезка пробелов по краям строки
            stripped = line.strip()
            # Начало блока критериев с talc_flaky:
            if stripped.startswith("talc_flaky:"):
                # Включить режим захвата строк
                capture = True
            # Если захват активен — накапливать строки
            if capture:
                # Остановка перед разделом «Рекомендации по разметке»
                if stripped.startswith("Рекомендации по разметке"):
                    # Выход из цикла построчного разбора
                    break
                # Добавление строки критерия в буфер
                criteria_lines.append(line)

        # Текстовый блок со списком всех 10 классов
        classes_block = "\n".join(f"{c['id']} — {c['name']}: {c['title']}" for c in CLASSES)
        # Текстовый блок критериев или запасной фрагмент test_c2.txt
        criteria_block = "\n".join(criteria_lines).strip() or criteria_source[:2500]

        # Подстановка плейсхолдеров в шаблон и обрезка пробелов
        prompt = (
            template.replace("{classes_block}", classes_block).replace("{criteria_block}", criteria_block).strip()
        )

        # Сводка результата шага 3
        result = {"step": 3, "prompt_chars": len(prompt), "classes": len(CLASSES), "prompt_preview": prompt[:500]}
        # Вывод JSON-отчёта шага 3
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Возврат готового промпта для классификации
        return prompt

    # Вспомогательная функция: кодирование изображения в base64 и MIME
    def _encode_image(image_path: Path) -> tuple[str, str]:
        # Определение MIME-типа по расширению файла
        mime, _ = mimetypes.guess_type(str(image_path))
        # Запасной MIME, если тип не определён
        if not mime:
            # По умолчанию считаем JPEG
            mime = "image/jpeg"
        # Чтение файла и кодирование в base64 ASCII-строку
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        # Возврат пары (mime, base64_data)
        return mime, data

    # Вспомогательная функция: разбор JSON-ответа модели в структуру классификации
    def _parse_classification(raw_text: str) -> dict[str, Any] | None:
        # Удаление лишних пробелов в начале и конце ответа
        text = raw_text.strip()
        # Пустой ответ — не удалось классифицировать
        if not text:
            # Возврат None при пустом тексте
            return None

        # Список кандидатов на парсинг JSON (исходный текст первым)
        candidates = [text]
        # Поиск JSON внутри markdown-блока ```json ... ```
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        # Если найден fenced JSON — вставить его первым кандидатом
        if fence:
            # Извлечение содержимого из группы регулярного выражения
            candidates.insert(0, fence.group(1))
        # Поиск первого JSON-объекта в произвольном тексте
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        # Если найден объект в фигурных скобках — добавить в кандидаты
        if brace:
            # Добавление найденного фрагмента
            candidates.append(brace.group(0))

        # Перебор всех кандидатов на валидный JSON
        for candidate in candidates:
            # Попытка десериализации JSON
            try:
                # Парсинг строки кандидата в Python-объект
                payload = json.loads(candidate)
            # Пропуск невалидного JSON
            except json.JSONDecodeError:
                # Переход к следующему кандидату
                continue
            # Ответ должен быть словарём
            if not isinstance(payload, dict):
                # Пропуск не-словаря
                continue

            # Извлечение числового class_id из ответа модели
            class_id = payload.get("class_id")
            # Извлечение строкового class_name из ответа модели
            class_name = str(payload.get("class_name", "")).strip()

            # Если id отсутствует — попытка восстановить по имени класса
            if class_id is None and class_name:
                # Поиск по полной метке вида 1_talc_flaky
                if class_name in CLASS_BY_NAME:
                    # Присвоение id из справочника CLASS_BY_NAME
                    class_id = CLASS_BY_NAME[class_name]["id"]
                # Иначе поиск по устаревшему короткому имени
                else:
                    # Нормализация имени: нижний регистр и пробелы → подчёркивания
                    normalized = class_name.lower().replace(" ", "_")
                    # Проверка в таблице LEGACY_NAME_MAP
                    if normalized in LEGACY_NAME_MAP:
                        # Присвоение id из устаревшей таблицы
                        class_id = LEGACY_NAME_MAP[normalized]

            # Приведение class_id к целому числу
            try:
                # Конвертация в int
                class_id = int(class_id)
            # Пропуск при невозможности конвертации
            except (TypeError, ValueError):
                # Переход к следующему кандидату
                continue

            # Проверка, что id входит в допустимый диапазон 0–9
            if class_id not in CLASS_BY_ID:
                # Пропуск неизвестного class_id
                continue

            # Успешный разбор — возврат нормализованной структуры
            return {
                # Числовой идентификатор класса
                "class_id": class_id,
                # Каноническое имя класса из CLASS_BY_ID
                "class_name": CLASS_BY_ID[class_id]["name"],
                # Уверенность модели (0.0 по умолчанию)
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
                # Краткое обоснование классификации на русском
                "rationale": str(payload.get("rationale", "")).strip(),
                # Исходный необработанный текст ответа модели
                "raw": raw_text,
            }
        # Ни один кандидат не дал валидного результата
        return None

    # Вспомогательная функция: один запрос классификации изображения через OpenRouter
    def _classify_image(config: dict[str, Any], prompt: str, image_path: Path) -> dict[str, Any] | None:
        # Кодирование изображения в base64 и определение MIME
        mime, image_b64 = _encode_image(image_path)
        # HTTP-заголовки запроса к OpenRouter
        headers = {
            # Авторизация Bearer с API-ключом
            "Authorization": f"Bearer {config['api_key']}",
            # Тип тела запроса — JSON
            "Content-Type": "application/json",
            # Рекомендуемый заголовок Referer для OpenRouter
            "HTTP-Referer": "https://github.com/colab_c2",
            # Название приложения для статистики OpenRouter
            "X-Title": "colab_c2_talc_sampler",
        }
        # Тело POST-запроса в формате OpenAI Chat Completions
        body = {
            # Идентификатор vision-модели DeepSeek
            "model": config["model"],
            # Низкая температура для стабильной классификации
            "temperature": 0.1,
            # Список сообщений диалога
            "messages": [
                # Единственное пользовательское сообщение с текстом и картинкой
                {
                    # Роль отправителя
                    "role": "user",
                    # Мультимодальное содержимое: промпт + изображение
                    "content": [
                        # Текстовая часть — инструкция и критерии классов
                        {"type": "text", "text": prompt},
                        # Визуальная часть — data URL с base64
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    ],
                }
            ],
        }

        # Текст последней ошибки для логирования при неудаче
        last_error = ""
        # Цикл повторных попыток при сбоях сети или API
        for attempt in range(1, RETRY_COUNT + 1):
            # Попытка выполнить HTTP POST
            try:
                # Отправка запроса к OpenRouter
                response = requests.post(
                    # URL эндпоинта chat/completions
                    OPENROUTER_URL,
                    # Заголовки с авторизацией
                    headers=headers,
                    # JSON-тело с моделью и сообщениями
                    json=body,
                    # Таймаут ожидания ответа
                    timeout=REQUEST_TIMEOUT,
                )
                # Обработка HTTP-ошибок 4xx/5xx
                if response.status_code >= 400:
                    # Сохранение текста ошибки для лога
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    # Пауза перед повтором с нарастающей задержкой
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    # Переход к следующей попытке
                    continue

                # Парсинг JSON-ответа OpenRouter
                payload = response.json()
                # Извлечение списка вариантов ответа модели
                choices = payload.get("choices") or []
                # Пустой choices — ошибка ответа
                if not choices:
                    # Фиксация причины неудачи
                    last_error = "Пустой ответ choices"
                    # Пауза перед повтором
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    # Переход к следующей попытке
                    continue

                # Текстовое содержимое первого варианта ответа
                content = choices[0].get("message", {}).get("content", "")
                # Некоторые модели возвращают content как список частей
                if isinstance(content, list):
                    # Склейка текстовых частей в одну строку
                    content = "\n".join(
                        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                    )
                # Разбор JSON-классификации из текста ответа
                parsed = _parse_classification(str(content))
                # Успешный разбор — вернуть результат
                if parsed:
                    # Возврат структуры классификации
                    return parsed
                # Не удалось распарсить — запомнить фрагмент ответа
                last_error = f"Не удалось распарсить ответ: {str(content)[:300]}"
            # Обработка сетевых и транспортных ошибок requests
            except requests.RequestException as exc:
                # Сохранение текста исключения
                last_error = str(exc)

            # Пауза перед следующей попыткой
            time.sleep(RETRY_DELAY_SEC * attempt)

        # Все попытки исчерпаны — вывести предупреждение в stdout
        print(json.dumps({"warning": "classification_failed", "image": str(image_path), "error": last_error}, ensure_ascii=False))
        # Возврат None — классификация не удалась
        return None

    # Шаг 4: классификация снимков и формирование сбалансированной выборки
    def step_4_classify_and_sample(
        config: dict[str, Any], images: list[Path], prompt: str
    ) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
        # Счётчик принятых снимков по каждому class_id
        counts = {c["id"]: 0 for c in CLASSES}
        # Выборка: списки записей по каждому class_id
        sample: dict[int, list[dict[str, Any]]] = {c["id"]: [] for c in CLASSES}
        # Полный журнал всех попыток классификации
        log: list[dict[str, Any]] = []
        # Целевая квота на класс (обычно 100)
        target = config["target_per_class"]

        # Вложенная функция: проверка заполнения всех квот
        def quota_full() -> bool:
            # True, если по каждому классу набрано не меньше target снимков
            return all(counts[cid] >= target for cid in counts)

        # Счётчик выполненных попыток классификации
        attempts = 0
        # Перебор изображений в случайном порядке
        for image_path in images:
            # Остановка при заполнении квот или достижении лимита попыток
            if quota_full() or attempts >= MAX_ATTEMPTS:
                # Выход из цикла по изображениям
                break
            # Увеличение счётчика попыток
            attempts += 1

            # Вызов OpenRouter для классификации текущего снимка
            prediction = _classify_image(config, prompt, image_path)
            # Пропуск при ошибке API или парсинга
            if prediction is None:
                # Запись неудачной попытки в журнал
                log.append({"image": str(image_path), "accepted": False, "reason": "parse_or_api_error"})
                # Переход к следующему изображению
                continue

            # Извлечение предсказанного class_id
            class_id = prediction["class_id"]
            # Принять снимок, только если квота класса ещё не заполнена
            accepted = counts[class_id] < target
            # Формирование записи о классификации
            record = {
                # Путь к исходному файлу
                "image": str(image_path),
                # Числовой класс
                "class_id": class_id,
                # Строковая метка класса
                "class_name": prediction["class_name"],
                # Уверенность модели
                "confidence": prediction["confidence"],
                # Краткое обоснование
                "rationale": prediction["rationale"],
                # Флаг включения в итоговую выборку
                "accepted": accepted,
            }
            # Добавление записи в общий журнал
            log.append(record)

            # Если снимок принят — обновить счётчик и выборку
            if accepted:
                # Увеличение счётчика для данного класса
                counts[class_id] += 1
                # Добавление записи в выборку класса
                sample[class_id].append(record)

        # Сводка результата шага 4
        result = {
            # Номер шага
            "step": 4,
            # Число выполненных попыток классификации
            "attempts": attempts,
            # Текущие счётчики по классам
            "counts": counts,
            # Суммарное число принятых снимков
            "total_selected": sum(counts.values()),
            # Флаг полного заполнения всех квот
            "quota_full": quota_full(),
        }
        # Вывод JSON-отчёта шага 4
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Возврат выборки и журнала классификаций
        return sample, log

    # Шаг 5: сохранение выборки на диск и запись manifest.json
    def step_5_save_sample(
        config: dict[str, Any], sample: dict[int, list[dict[str, Any]]], log: list[dict[str, Any]]
    ) -> Path:
        # Путь к каталогу output из конфигурации
        output_dir: Path = config["output_dir"]
        # Путь к каталогу sample из конфигурации
        sample_dir: Path = config["sample_dir"]
        # Путь к manifest.json из конфигурации
        manifest_file: Path = config["manifest_file"]

        # Удаление предыдущей выборки, если каталог уже существует
        if sample_dir.exists():
            # Рекурсивное удаление старого sample/
            shutil.rmtree(sample_dir)
        # Создание пустого каталога sample с родителями
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Список строк манифеста для итогового JSON
        manifest_rows: list[dict[str, Any]] = []
        # Обход классов в порядке возрастания id
        for class_id in sorted(sample.keys()):
            # Каноническое имя каталога класса
            class_name = CLASS_BY_ID[class_id]["name"]
            # Подкаталог sample/<class_name>/
            class_dir = sample_dir / class_name
            # Создание подкаталога класса
            class_dir.mkdir(parents=True, exist_ok=True)

            # Копирование каждого принятого снимка класса
            for idx, item in enumerate(sample[class_id], start=1):
                # Исходный путь к файлу из записи
                src = Path(item["image"])
                # Имя файла в выборке: порядковый номер + исходное имя
                dst_name = f"{idx:03d}_{src.stem}{src.suffix.lower()}"
                # Полный путь назначения в sample/<class>/
                dst = class_dir / dst_name
                # Копирование с сохранением метаданных файла
                shutil.copy2(src, dst)
                # Строка манифеста для одного снимка
                row = {
                    # Числовой id класса
                    "class_id": class_id,
                    # Строковая метка класса
                    "class_name": class_name,
                    # Путь к исходному файлу
                    "source_image": str(src),
                    # Путь к копии в выборке
                    "sample_image": str(dst),
                    # Уверенность модели при классификации
                    "confidence": item.get("confidence"),
                    # Обоснование модели
                    "rationale": item.get("rationale"),
                }
                # Добавление строки в список манифеста
                manifest_rows.append(row)

        # Создание каталога output, если его ещё нет
        output_dir.mkdir(parents=True, exist_ok=True)
        # Сборка полного объекта манифеста
        manifest = {
            # Модель, использованная для классификации
            "model": config["model"],
            # Целевая квота на класс
            "target_per_class": config["target_per_class"],
            # Фактические счётчики по именам классов
            "counts": {CLASS_BY_ID[cid]["name"]: len(sample[cid]) for cid in sorted(sample)},
            # Список всех сохранённых элементов выборки
            "items": manifest_rows,
            # Полный журнал всех попыток классификации
            "classification_log": log,
        }
        # Запись manifest.json на диск в UTF-8
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # Сводка результата шага 5
        result = {
            # Номер шага
            "step": 5,
            # Путь к записанному manifest.json
            "manifest": str(manifest_file),
            # Путь к каталогу sample
            "sample_dir": str(sample_dir),
            # Число сохранённых изображений
            "saved_images": len(manifest_rows),
            # Распределение по классам
            "counts": manifest["counts"],
        }
        # Вывод JSON-отчёта шага 5
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # Возврат пути к manifest.json
        return manifest_file

    # Вызов шага 1: загрузка конфигурации
    config = step_1_load_config()
    # Вызов шага 2: сбор списка изображений
    images = step_2_collect_images(config)
    # Вызов шага 3: построение промпта
    prompt = step_3_build_prompt(config)
    # Вызов шага 4: классификация и формирование выборки
    sample, log = step_4_classify_and_sample(config, images, prompt)
    # Вызов шага 5: сохранение результатов на диск
    step_5_save_sample(config, sample, log)


# Точка входа при запуске файла как скрипта
if __name__ == "__main__":
    # Запуск главной функции main()
    main()
