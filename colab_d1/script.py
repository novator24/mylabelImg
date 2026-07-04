# Шебанг: интерпретатор Python 3 для Unix-систем
#!/usr/bin/env python3
# Директива кодировки исходного файла UTF-8
# -*- coding: utf-8 -*-
# Модульная документация: назначение скрипта — детекция YOLO
"""Детекция металлического, кремового блеска и прочего с помощью YOLO."""

# Импорт отложенной оценки аннотаций типов
from __future__ import annotations

# Импорт модуля генерации псевдослучайных чисел
import random
# Импорт утилит высокоуровневого копирования файлов и каталогов
import shutil
# Импорт класса Path для работы с путями файловой системы
from pathlib import Path

# Пустая строка-разделитель групп импортов

# Импорт OpenCV для операций с изображениями
import cv2
# Импорт NumPy для работы с массивами
import numpy as np
# Импорт PyYAML для записи конфигурации датасета
import yaml
# Импорт Pillow Image для загрузки и трансформации изображений
from PIL import Image


# Пустая строка-разделитель перед константами путей

# Корень проекта — родитель каталога colab_d1
ROOT = Path(__file__).resolve().parent.parent
# Каталог colab_d1, где лежит этот скрипт
COLAB = Path(__file__).resolve().parent
# Путь к исходному металлическому градиенту test_d1.png
SOURCE_PNG = ROOT / "test_d1.png"
# Путь к исходному кремовому градиенту test_d1.jpg
SOURCE_JPG = ROOT / "test_d1.jpg"
# Каталог для сгенерированных металлических оверлеев
OVERLAYS_PNG = COLAB / "overlays" / "png"
# Каталог для сгенерированных кремовых оверлеев
OVERLAYS_JPG = COLAB / "overlays" / "jpg"
# Каталог YOLO-датасета (images, labels, data.yaml)
DATASET = COLAB / "dataset"
# Каталог сохранения обученной модели и весов
RESULT = COLAB / "result"

# Имена классов YOLO: 0 — other, 1 — tonkie, 2 — alter
CLASS_NAMES = ("other", "tonkie", "alter")
# Имена целевых каталогов с исходными изображениями
TARGET_DIRS = ("0_other", "1_tonkie", "2_alter")
# Допустимые расширения файлов изображений при рекурсивном поиске
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Количество аугментированных оверлеев на каждый исходник
OVERLAY_COUNT = 1000
# Фиксированный генератор случайных чисел для воспроизводимости
RNG = random.Random(42)


# Пустая строка перед объявлением главной функции

# Главная функция пайплайна без аргументов, возвращает None
def main() -> None:
    # Docstring: описание полного цикла работы main
    """Запуск полного пайплайна: аугментация, датасет, обучение YOLO."""

    # Вложенная функция step_1: генерация прозрачных оверлеев
    def step_1_generate_overlays() -> dict:
        # Docstring step_1: параметры аугментации оверлеев
        """1000 прозрачных оверлеев из test_d1.png и test_d1.jpg (поворот 1–90°, масштаб x2–x5)."""

        # Внутренняя функция создания пула аугментированных изображений
        def _make_pool(source: Path, out_dir: Path, prefix: str) -> list[Path]:
            # Создание выходного каталога, включая родительские пути
            out_dir.mkdir(parents=True, exist_ok=True)
            # Загрузка исходника и конвертация в RGBA с альфа-каналом
            img = Image.open(source).convert("RGBA")
            # Список путей к сохранённым оверлеям
            paths: list[Path] = []
            # Цикл генерации OVERLAY_COUNT аугментированных вариантов
            for i in range(OVERLAY_COUNT):
                # Случайный угол поворота от 1 до 90 градусов
                angle = RNG.uniform(1.0, 90.0)
                # Случайный масштаб от 2x до 5x
                scale = RNG.uniform(2.0, 5.0)
                # Получение ширины и высоты исходного изображения
                w, h = img.size
                # Масштабирование изображения с интерполяцией LANCZOS
                scaled = img.resize(
                    # Новые размеры с защитой от нулевой ширины/высоты
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    # Метод ресемплинга высокого качества
                    Image.Resampling.LANCZOS,
                # Закрытие вызова resize
                )
                # Поворот масштабированного изображения с расширением холста
                rotated = scaled.rotate(
                    # Угол поворота в градусах
                    angle, expand=True, resample=Image.Resampling.BICUBIC
                # Закрытие вызова rotate
                )
                # Формирование пути сохранения с нумерацией и префиксом
                path = out_dir / f"{prefix}_{i:04d}.png"
                # Запись PNG с прозрачным фоном на диск
                rotated.save(path)
                # Добавление пути в список результатов
                paths.append(path)
            # Возврат списка путей ко всем сгенерированным оверлеям
            return paths

        # Генерация пула металлических оверлеев из test_d1.png
        png_paths = _make_pool(SOURCE_PNG, OVERLAYS_PNG, "metal")
        # Генерация пула кремовых оверлеев из test_d1.jpg
        jpg_paths = _make_pool(SOURCE_JPG, OVERLAYS_JPG, "cream")
        # Словарь-результат step_1 с количеством и путями
        result = {
            # Число сгенерированных PNG-оверлеев
            "png_count": len(png_paths),
            # Число сгенерированных JPG-оверлеев (сохранены как PNG)
            "jpg_count": len(jpg_paths),
            # Строковый путь к каталогу металлических оверлеев
            "png_dir": str(OVERLAYS_PNG),
            # Строковый путь к каталогу кремовых оверлеев
            "jpg_dir": str(OVERLAYS_JPG),
        # Закрытие словаря result
        }
        # Вывод результата step_1 в консоль
        print(f"[step_1] {result}")
        # Возврат словаря результата вызывающему коду
        return result

    # Вложенная функция step_2: сбор исходных изображений из каталогов
    def step_2_collect_source_images() -> dict:
        # Docstring step_2: рекурсивный поиск jpg/png
        """Рекурсивный поиск jpg/png в каталогах 0_other, 1_tonkie, 2_alter."""

        # Внутренняя функция сканирования одного именованного каталога
        def _scan(name: str) -> list[Path]:
            # Аккумулятор найденных путей к изображениям
            found: list[Path] = []
            # Обход корня проекта и каталога colab_d1
            for base in (ROOT, COLAB):
                # Рекурсивный поиск папок с заданным именем
                for folder in base.rglob(name):
                    # Пропуск, если найденный объект не является каталогом
                    if not folder.is_dir():
                        # Переход к следующему совпадению rglob
                        continue
                    # Рекурсивный обход всех файлов внутри найденного каталога
                    for path in folder.rglob("*"):
                        # Проверка: файл с допустимым расширением изображения
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                            # Добавление абсолютного пути в список
                            found.append(path.resolve())
            # Возврат отсортированного списка уникальных путей
            return sorted(set(found))

        # Словарь: имя каталога → список найденных изображений
        images = {name: _scan(name) for name in TARGET_DIRS}
        # Словарь: имя каталога → количество найденных изображений
        result = {name: len(paths) for name, paths in images.items()}
        # Вывод результата step_2 в консоль
        print(f"[step_2] {result}")
        # Возврат полного словаря с путями и счётчиками
        return {"images": images, "counts": result}

    # Вспомогательная функция: подготовка структуры каталогов датасета
    def _init_dataset_dirs() -> None:
        # Перебор подкаталогов train/val для images и labels
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            # Полный путь к подкаталогу внутри DATASET
            target = DATASET / sub
            # Удаление существующего каталога для чистой пересборки
            if target.exists():
                # Рекурсивное удаление каталога со всем содержимым
                shutil.rmtree(target)
            # Создание пустого каталога (и родителей при необходимости)
            target.mkdir(parents=True, exist_ok=True)

    # Вспомогательная функция: перевод BGR-изображения в чёрно-белое RGB
    def _to_bw_rgb(arr: np.ndarray) -> np.ndarray:
        # Конвертация BGR → оттенки серого
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        # Конвертация GRAY → BGR (три одинаковых канала для совместимости)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Вспомогательная функция: bounding box по непрозрачным пикселям альфа-маски
    def _overlay_bbox(alpha: np.ndarray, x0: int, y0: int) -> tuple[int, int, int, int] | None:
        # Координаты пикселей с альфой выше порога 10
        ys, xs = np.where(alpha > 10)
        # Если непрозрачных пикселей нет — bbox не определяется
        if len(xs) == 0:
            # Возврат None сигнализирует об отсутствии метки
            return None
        # Возврат bbox (x1, y1, x2, y2) в координатах базового изображения
        return (
            # Левая граница bbox с учётом смещения x0
            int(xs.min() + x0),
            # Верхняя граница bbox с учётом смещения y0
            int(ys.min() + y0),
            # Правая граница bbox с учётом смещения x0
            int(xs.max() + x0),
            # Нижняя граница bbox с учётом смещения y0
            int(ys.max() + y0),
        # Закрытие кортежа bbox
        )

    # Вспомогательная функция: наложение полупрозрачного оверлея на базу
    def _place_overlay(
        # Базовое изображение в формате BGR NumPy-массива
        base: np.ndarray,
        # Путь к файлу оверлея с альфа-каналом
        overlay_path: Path,
        # Коэффициент полупрозрачности оверлея (0..1)
        alpha_scale: float,
    # Аннотация возвращаемого типа: база и опциональный bbox
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        # Высота и ширина базового изображения
        h, w = base.shape[:2]
        # Загрузка оверлея и конвертация в RGBA
        ov = Image.open(overlay_path).convert("RGBA")
        # Ширина и высота оверлея
        ow, oh = ov.size
        # Если оверлей больше базы — уменьшить до вписывания
        if ow > w or oh > h:
            # Коэффициент масштабирования с небольшим случайным запасом
            ratio = min(w / ow, h / oh) * RNG.uniform(0.5, 0.95)
            # Ресайз оверлея под размер базового изображения
            ov = ov.resize(
                # Новые размеры с защитой от нуля
                (max(1, int(ow * ratio)), max(1, int(oh * ratio))),
                # Интерполяция LANCZOS для качества
                Image.Resampling.LANCZOS,
            # Закрытие вызова resize
            )
            # Обновление размеров оверлея после ресайза
            ow, oh = ov.size
        # Случайная координата X левого верхнего угла оверлея
        x0 = RNG.randint(0, max(0, w - ow))
        # Случайная координата Y левого верхнего угла оверлея
        y0 = RNG.randint(0, max(0, h - oh))
        # Конвертация PIL-изображения оверлея в NumPy-массив
        ov_arr = np.array(ov)
        # Нормализованный альфа-канал с учётом alpha_scale
        alpha = (ov_arr[:, :, 3].astype(np.float32) / 255.0) * alpha_scale
        # RGB-каналы оверлея в BGR-порядке OpenCV
        rgb = ov_arr[:, :, :3][:, :, ::-1].astype(np.float32)
        # Вырезка области базы под оверлей, приведение к float32
        region = base[y0 : y0 + oh, x0 : x0 + ow].astype(np.float32)
        # Альфа-блендинг: база × (1−α) + оверлей × α
        blended = region * (1.0 - alpha[..., None]) + rgb * alpha[..., None]
        # Запись смешанной области обратно в базовое изображение
        base[y0 : y0 + oh, x0 : x0 + ow] = blended.astype(np.uint8)
        # Вычисление bbox по маске непрозрачности
        bbox = _overlay_bbox((alpha * 255).astype(np.uint8), x0, y0)
        # Возврат изменённой базы и bbox (или None)
        return base, bbox

    # Вспомогательная функция: конвертация bbox пикселей в формат YOLO
    def _bbox_to_yolo(
        # Bounding box в абсолютных пикселях (x1, y1, x2, y2)
        bbox: tuple[int, int, int, int], img_w: int, img_h: int
    # Возвращает нормализованные xc, yc, w, h для YOLO
    ) -> tuple[float, float, float, float]:
        # Распаковка координат bbox
        x1, y1, x2, y2 = bbox
        # Нормализованная X-координата центра bbox
        xc = ((x1 + x2) / 2.0) / img_w
        # Нормализованная Y-координата центра bbox
        yc = ((y1 + y2) / 2.0) / img_h
        # Нормализованная ширина bbox
        bw = (x2 - x1) / img_w
        # Нормализованная высота bbox
        bh = (y2 - y1) / img_h
        # Возврат кортежа YOLO-координат
        return xc, yc, bw, bh

    # Общий список всех сгенерированных обучающих образцов пайплайна
    all_samples: list[
        # Элемент: (имя_файла, изображение, список меток YOLO)
        tuple[str, np.ndarray, list[tuple[int, tuple[float, float, float, float]]]]
    # Инициализация пустым списком
    ] = []

    # Вспомогательная функция: обработка одной категории изображений
    def _process_category(
        # Список путей к исходным изображениям категории
        paths: list[Path],
        # Спецификации оверлеев: (пул_файлов, id_класса YOLO)
        overlay_specs: list[tuple[list[Path], int]],
        # Тег категории для формирования имён файлов
        tag: str,
    # Возвращает словарь со статистикой обработки
    ) -> dict:
        # Счётчик успешно обработанных изображений в категории
        count = 0
        # Перебор каждого исходного изображения категории
        for src in paths:
            # Чтение изображения через OpenCV (BGR)
            raw = cv2.imread(str(src))
            # Пропуск файла, если OpenCV не смог его загрузить
            if raw is None:
                # Переход к следующему исходнику
                continue
            # Преобразование исходника в чёрно-белое RGB-представление
            base = _to_bw_rgb(raw)
            # Список меток YOLO для текущего образца
            labels: list[tuple[int, tuple[float, float, float, float]]] = []
            # Перебор пар (пул оверлеев, id класса)
            for pool, cls_id in overlay_specs:
                # Пропуск, если пул оверлеев пуст
                if not pool:
                    # Переход к следующей спецификации оверлея
                    continue
                # Случайный выбор одного оверлея из пула
                ov_path = RNG.choice(pool)
                # Случайная степень полупрозрачности оверлея
                alpha = RNG.uniform(0.35, 0.75)
                # Наложение оверлея и получение bbox
                base, bbox = _place_overlay(base, ov_path, alpha)
                # Пропуск метки, если bbox не определён
                if bbox is None:
                    # Переход к следующему оверлею
                    continue
                # Размеры текущего базового изображения после наложения
                h, w = base.shape[:2]
                # Добавление метки YOLO (класс + нормализованный bbox)
                labels.append((cls_id, _bbox_to_yolo(bbox, w, h)))
            # Пропуск образца, если не удалось создать ни одной метки
            if not labels:
                # Переход к следующему исходному изображению
                continue
            # Уникальное имя файла образца с тегом категории и порядковым номером
            stem = f"{tag}_{src.stem}_{len(all_samples):06d}"
            # Добавление образца в общий список all_samples
            all_samples.append((stem, base, labels))
            # Увеличение счётчика обработанных изображений категории
            count += 1
        # Возврат словаря с тегом, числом обработанных и общим числом образцов
        return {"tag": tag, "processed": count, "total_samples": len(all_samples)}

    # Вложенная функция step_3: обработка каталога 1_tonkie
    def step_3_process_tonkie(sources: dict, overlays: dict) -> dict:
        # Docstring step_3: металлический оверлей, YOLO class 1
        """1_tonkie: B&W + полупрозрачный test_d1.png, YOLO class 1."""

        # Отсортированный список металлических оверлеев PNG
        png_pool = sorted(OVERLAYS_PNG.glob("*.png"))
        # Обработка изображений 1_tonkie с классом 1
        result = _process_category(
            # Пути изображений категории 1_tonkie
            sources["images"]["1_tonkie"], [(png_pool, 1)], "1_tonkie"
        # Закрытие вызова _process_category
        )
        # Вывод результата step_3 в консоль
        print(f"[step_3] {result}")
        # Возврат словаря результата
        return result

    # Вложенная функция step_4: обработка каталога 2_alter
    def step_4_process_alter(sources: dict, overlays: dict) -> dict:
        # Docstring step_4: кремовый оверлей, YOLO class 2
        """2_alter: B&W + полупрозрачный test_d1.jpg, YOLO class 2."""

        # Отсортированный список кремовых оверлеев (сохранены как PNG)
        jpg_pool = sorted(OVERLAYS_JPG.glob("*.png"))
        # Обработка изображений 2_alter с классом 2
        result = _process_category(
            # Пути изображений категории 2_alter
            sources["images"]["2_alter"], [(jpg_pool, 2)], "2_alter"
        # Закрытие вызова _process_category
        )
        # Вывод результата step_4 в консоль
        print(f"[step_4] {result}")
        # Возврат словаря результата
        return result

    # Вложенная функция step_5: обработка каталога 0_other
    def step_5_process_other(sources: dict, overlays: dict) -> dict:
        # Docstring step_5: два оверлея, оба YOLO class 0
        """0_other: B&W + два полупрозрачных оверлея (png + jpg), YOLO class 0."""

        # Пул металлических оверлеев для класса 0
        png_pool = sorted(OVERLAYS_PNG.glob("*.png"))
        # Пул кремовых оверлеев для класса 0
        jpg_pool = sorted(OVERLAYS_JPG.glob("*.png"))
        # Обработка 0_other с двумя оверлеями, оба класс 0
        result = _process_category(
            # Пути изображений категории 0_other
            sources["images"]["0_other"],
            # Два оверлея: металл и крем, оба class 0
            [(png_pool, 0), (jpg_pool, 0)],
            # Тег категории для имён файлов
            "0_other",
        # Закрытие вызова _process_category
        )
        # Вывод результата step_5 в консоль
        print(f"[step_5] {result}")
        # Возврат словаря результата
        return result

    # Вложенная функция step_6: сохранение датасета и data.yaml
    def step_6_save_dataset_and_yaml() -> dict:
        # Docstring step_6: train/val сплит и конфигурация YOLO
        """Сохранение train/val сплита и data.yaml."""

        # Пересоздание структуры каталогов датасета
        _init_dataset_dirs()
        # Проверка: есть ли хотя бы один сгенерированный образец
        if not all_samples:
            # Словарь-результат для пустого датасета
            result = {
                # Общее число образцов — ноль
                "total": 0,
                # Число train-образцов — ноль
                "train": 0,
                # Число val-образцов — ноль
                "val": 0,
                # Путь к каталогу датасета
                "dataset_dir": str(DATASET),
                # Предупреждение об отсутствии исходных изображений
                "warning": "нет исходных изображений в 0_other/1_tonkie/2_alter",
            # Закрытие словаря result
            }
            # Вывод предупреждения step_6 в консоль
            print(f"[step_6] {result}")
            # Досрочный возврат без сохранения файлов
            return result

        # Копия списка образцов для перемешивания
        shuffled = list(all_samples)
        # Случайное перемешивание образцов перед разбиением
        RNG.shuffle(shuffled)
        # Размер val-выборки: 20% от общего числа, минимум 1
        val_count = max(1, int(len(shuffled) * 0.2))
        # Словарь сплитов: train — основная часть, val — первые val_count
        splits = {"train": shuffled[val_count:], "val": shuffled[:val_count]}

        # Перебор сплитов train и val
        for split, items in splits.items():
            # Перебор образцов внутри текущего сплита
            for stem, image, labels in items:
                # Путь сохранения JPEG-изображения образца
                img_path = DATASET / "images" / split / f"{stem}.jpg"
                # Путь сохранения YOLO-метки образца
                lbl_path = DATASET / "labels" / split / f"{stem}.txt"
                # Запись изображения на диск через OpenCV
                cv2.imwrite(str(img_path), image)
                # Открытие файла метки для записи в UTF-8
                with lbl_path.open("w", encoding="utf-8") as f:
                    # Запись каждой строки метки YOLO
                    for cls_id, (xc, yc, bw, bh) in labels:
                        # Строка формата: class x_center y_center width height
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        # Словарь конфигурации датасета для Ultralytics YOLO
        data_yaml = {
            # Абсолютный путь к корню датасета
            "path": str(DATASET.resolve()),
            # Относительный путь к train-изображениям
            "train": "images/train",
            # Относительный путь к val-изображениям
            "val": "images/val",
            # Словарь id → имя класса
            "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        # Закрытие словаря data_yaml
        }
        # Путь к файлу data.yaml
        yaml_path = DATASET / "data.yaml"
        # Запись data.yaml на диск
        with yaml_path.open("w", encoding="utf-8") as f:
            # Сериализация словаря в YAML с поддержкой Unicode
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

        # Словарь-результат step_6 после успешного сохранения
        result = {
            # Общее число образцов в датасете
            "total": len(shuffled),
            # Число образцов в train-сплите
            "train": len(splits["train"]),
            # Число образцов в val-сплите
            "val": len(splits["val"]),
            # Путь к каталогу датасета
            "dataset_dir": str(DATASET),
            # Путь к файлу data.yaml
            "data_yaml": str(yaml_path),
        # Закрытие словаря result
        }
        # Вывод результата step_6 в консоль
        print(f"[step_6] {result}")
        # Возврат словаря результата
        return result

    # Вложенная функция step_7: обучение YOLO и сохранение весов
    def step_7_train_yolo(dataset_info: dict) -> dict:
        # Docstring step_7: обучение YOLOv8, сохранение в result/
        """Обучение YOLOv8, сохранение весов в result/."""

        # Проверка: датасет пуст — обучение пропускается
        if dataset_info.get("total", 0) == 0:
            # Словарь-результат для пропущенного обучения
            result = {
                # Статус: обучение не выполнялось
                "status": "skipped",
                # Причина пропуска — пустой датасет
                "reason": "пустой датасет — добавьте изображения в 0_other, 1_tonkie, 2_alter",
                # Путь к каталогу result
                "result_dir": str(RESULT),
            # Закрытие словаря result
            }
            # Вывод причины пропуска step_7 в консоль
            print(f"[step_7] {result}")
            # Досрочный возврат без обучения
            return result

        # Ленивый импорт Ultralytics YOLO (тяжёлая зависимость)
        from ultralytics import YOLO

        # Удаление предыдущего каталога result при повторном запуске
        if RESULT.exists():
            # Рекурсивное удаление старых весов и логов
            shutil.rmtree(RESULT)
        # Создание чистого каталога result
        RESULT.mkdir(parents=True, exist_ok=True)

        # Загрузка предобученной модели YOLOv8 small
        model = YOLO("yolov8s.pt")
        # Словарь гиперпараметров обучения с оптимальными настройками
        train_args = dict(
            # Путь к конфигурации датасета data.yaml
            data=str(DATASET / "data.yaml"),
            # Число эпох обучения
            epochs=150,
            # Размер входного изображения для сети
            imgsz=640,
            # Автовыбор batch size по доступной памяти GPU
            batch=-1,
            # Early stopping: терпимость к отсутствию улучшения mAP
            patience=25,
            # Оптимизатор AdamW для стабильной сходимости
            optimizer="AdamW",
            # Начальная скорость обучения
            lr0=0.001,
            # Конечная скорость обучения (доля от lr0)
            lrf=0.01,
            # Коэффициент момента SGD/Adam
            momentum=0.937,
            # L2-регуляризация (weight decay)
            weight_decay=0.0005,
            # Число эпох прогрева learning rate
            warmup_epochs=3,
            # Косинусный scheduler для learning rate
            cos_lr=True,
            # Аугментация: случайный сдвиг оттенка HSV
            hsv_h=0.015,
            # Аугментация: случайное изменение насыщенности
            hsv_s=0.7,
            # Аугментация: случайное изменение яркости
            hsv_v=0.4,
            # Аугментация: случайный поворот до ±10°
            degrees=10.0,
            # Аугментация: случайный сдвиг изображения
            translate=0.1,
            # Аугментация: случайное масштабирование
            scale=0.5,
            # Аугментация: горизонтальное отражение с вероятностью 0.5
            fliplr=0.5,
            # Аугментация: mosaic из 4 изображений
            mosaic=1.0,
            # Аугментация: mixup смешивание двух изображений
            mixup=0.1,
            # Аугментация: copy-paste объектов между изображениями
            copy_paste=0.1,
            # Корневой каталог проекта для сохранения результатов обучения
            project=str(RESULT),
            # Имя подкаталога эксперимента обучения
            name="train",
            # Разрешение перезаписи существующего эксперимента
            exist_ok=True,
            # Сохранение чекпоинтов модели
            save=True,
            # Генерация графиков метрик обучения
            plots=True,
            # Подробный вывод лога обучения в консоль
            verbose=True,
        # Закрытие словаря train_args
        )
        # Запуск обучения модели с заданными гиперпараметрами
        metrics = model.train(**train_args)

        # Каталог с весами после обучения
        weights_dir = RESULT / "train" / "weights"
        # Путь к лучшему чекпоинту по val mAP
        best_pt = weights_dir / "best.pt"
        # Путь к последнему чекпоинту
        last_pt = weights_dir / "last.pt"
        # Копирование best.pt и last.pt в корень result/
        for dst_name, src in (("best.pt", best_pt), ("last.pt", last_pt)):
            # Копировать только если исходный файл существует
            if src.exists():
                # Копирование с сохранением метаданных файла
                shutil.copy2(src, RESULT / dst_name)

        # Переменная для метрики mAP@0.5, по умолчанию None
        map50 = None
        # Извлечение mAP50 из results_dict, если metrics доступен
        if metrics is not None and hasattr(metrics, "results_dict"):
            # Чтение метрики metrics/mAP50(B) из словаря результатов
            map50 = metrics.results_dict.get("metrics/mAP50(B)", None)

        # Словарь-результат step_7 после успешного обучения
        result = {
            # Статус: обучение завершено
            "status": "done",
            # Путь к best.pt в result/, если файл существует
            "best_weights": str(RESULT / "best.pt") if (RESULT / "best.pt").exists() else None,
            # Путь к last.pt в result/, если файл существует
            "last_weights": str(RESULT / "last.pt") if (RESULT / "last.pt").exists() else None,
            # Значение mAP@0.5 на валидации (или None)
            "metrics_map50": map50,
            # Путь к каталогу result
            "result_dir": str(RESULT),
        # Закрытие словаря result
        }
        # Вывод результата step_7 в консоль
        print(f"[step_7] {result}")
        # Возврат словаря результата обучения
        return result

    # Вызов step_1: генерация аугментированных оверлеев
    overlays = step_1_generate_overlays()
    # Вызов step_2: сбор исходных изображений из каталогов
    sources = step_2_collect_source_images()
    # Вызов step_3: обработка категории 1_tonkie
    tonkie = step_3_process_tonkie(sources, overlays)
    # Вызов step_4: обработка категории 2_alter
    alter = step_4_process_alter(sources, overlays)
    # Вызов step_5: обработка категории 0_other
    other = step_5_process_other(sources, overlays)
    # Вызов step_6: сохранение датасета и data.yaml
    dataset = step_6_save_dataset_and_yaml()
    # Вызов step_7: обучение YOLO-модели
    training = step_7_train_yolo(dataset)

    # Итоговый сводный словарь результатов всех шагов
    summary = {
        # Результаты генерации оверлеев
        "overlays": overlays,
        # Счётчики найденных исходных изображений по каталогам
        "sources": sources["counts"],
        # Результат обработки 1_tonkie
        "tonkie": tonkie,
        # Результат обработки 2_alter
        "alter": alter,
        # Результат обработки 0_other
        "other": other,
        # Результат сборки и сохранения датасета
        "dataset": dataset,
        # Результат обучения модели
        "training": training,
    # Закрытие словаря summary
    }
    # Финальный вывод сводки пайплайна в консоль
    print(f"[main] finished: {summary}")


# Точка входа: запуск main() только при прямом вызове скрипта
if __name__ == "__main__":
    # Вызов главной функции пайплайна
    main()
