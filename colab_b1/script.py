"""Детекция текста с помощью SAM2 на изображении test_b1.png."""

# Включаем отложенную оценку аннотаций типов для совместимости с Python 3.7+
from __future__ import annotations

# Модуль для сериализации результатов шагов в формат JSON
import json
# Модуль для работы с путями к файлам и каталогам
from pathlib import Path

# OpenCV — обработка изображений, поиск контуров и отрисовка рамок
import cv2
# NumPy — работа с массивами пикселей и числовыми вычислениями
import numpy as np
# PyTorch — вычисления на GPU/CPU для нейросети SAM2
import torch
# Pillow — загрузка и сохранение растровых изображений
from PIL import Image
# Классы модели и препроцессора SAM2 из библиотеки Hugging Face Transformers
from transformers import Sam2Model, Sam2Processor


# Эталонный розовый цвет RGB, к которому ищем ближайший цвет на изображении
PINK_REFERENCE = np.array([255, 192, 203], dtype=np.float32)
# Допустимое евклидово расстояние в RGB-пространстве для совпадения с основным цветом
COLOR_TOLERANCE = 30
# Минимальная ширина вертикальной полосы, чтобы считать её текстовой зоной
MIN_TEXT_ZONE_WIDTH = 15


# Главная точка входа: последовательно выполняет все шаги пайплайна
def main() -> None:
    # Корень проекта — родительский каталог относительно colab_b1/script.py
    root = Path(__file__).resolve().parent.parent
    # Путь к исходному тестовому изображению в корне репозитория
    image_path = root / "test_b1.png"
    # Каталог для сохранения итогового изображения и рамок с текстом
    result_dir = root / "result"

    # Шаг 1: загрузка изображения test_b1.png в оперативную память
    def step_1_load_image() -> tuple[np.ndarray, Image.Image]:
        """Загрузить test_b1.png в память."""
        # Проверяем, что файл изображения существует по указанному пути
        if not image_path.exists():
            # Прерываем выполнение с понятным сообщением об ошибке
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        # Открываем PNG и приводим к трёхканальному формату RGB
        pil_image = Image.open(image_path).convert("RGB")
        # Конвертируем PIL-изображение в NumPy-массив формы (высота, ширина, 3)
        image = np.array(pil_image)
        # Формируем словарь с метаданными загруженного изображения для вывода
        result = {
            "path": str(image_path),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        }
        # Печатаем результат шага 1 в формате JSON с поддержкой кириллицы
        print("step_1_load_image:", json.dumps(result, ensure_ascii=False))
        # Возвращаем массив пикселей и объект PIL для дальнейших шагов
        return image, pil_image

    # Шаг 2: по вертикальным линиям определяем самую правую зону с текстом
    def step_2_find_rightmost_text_zone(image: np.ndarray) -> tuple[int, int, int, int]:
        """По вертикальным линиям найти самую правую зону с текстом."""
        # Получаем высоту и ширину изображения в пикселях
        height, width = image.shape[:2]
        # Переводим RGB-изображение в одноканальную шкалу серого
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Выделяем границы объектов оператором Кэнни для поиска текста
        edges = cv2.Canny(gray, 50, 150)
        # Маска тёмных пикселей — типичный признак символов текста
        dark = (gray < 120).astype(np.float32)

        # Массив оценок «текстовости» для каждой вертикальной колонки
        column_scores = np.zeros(width, dtype=np.float32)
        # Проходим по всем вертикальным линиям (столбцам) изображения
        for x in range(width):
            # Суммируем плотность краёв и тёмных пикселей в данном столбце
            column_scores[x] = edges[:, x].mean() + dark[:, x].mean() * 80.0

        # Ядро скользящего среднего для сглаживания шумных скачков по столбцам
        kernel = np.ones(7, dtype=np.float32) / 7.0
        # Сглаженный профиль текстовой активности вдоль оси X
        smooth_scores = np.convolve(column_scores, kernel, mode="same")
        # Порог отсечения: 28 % от максимума сглаженного профиля
        threshold = smooth_scores.max() * 0.28

        # Список найденных непрерывных текстовых зон в виде пар (x_start, x_end)
        zones: list[tuple[int, int]] = []
        # Флаг нахождения внутри текущей текстовой зоны при сканировании
        in_zone = False
        # Координата начала текущей зоны
        start = 0
        # Сканируем все столбцы слева направо для поиска зон с текстом
        for x in range(width):
            # Начало новой зоны: оценка выше порога, ранее зоны не было
            if smooth_scores[x] >= threshold and not in_zone:
                start = x
                in_zone = True
            # Конец зоны: оценка упала ниже порога, ранее были внутри зоны
            elif smooth_scores[x] < threshold and in_zone:
                # Добавляем зону только если её ширина не меньше минимальной
                if x - start >= MIN_TEXT_ZONE_WIDTH:
                    zones.append((start, x - 1))
                in_zone = False
        # Обрабатываем случай, когда зона доходит до правого края изображения
        if in_zone and width - start >= MIN_TEXT_ZONE_WIDTH:
            zones.append((start, width - 1))

        # Если ни одна зона не найдена — используем правую четверть изображения
        if not zones:
            zone = (int(width * 0.75), 0, width - 1, height - 1)
        else:
            # Выбираем зону с наибольшей координатой начала — самую правую
            x_start, x_end = max(zones, key=lambda zone_bounds: zone_bounds[0])
            # Формируем прямоугольник зоны на всю высоту изображения
            zone = (x_start, 0, x_end, height - 1)

        # Собираем результат шага: все зоны и выбранная правая зона
        result = {
            "all_text_zones": [list(zone_bounds) for zone_bounds in zones],
            "rightmost_text_zone": list(zone),
        }
        # Выводим результат шага 2 в консоль
        print("step_2_find_rightmost_text_zone:", json.dumps(result, ensure_ascii=False))
        # Возвращаем координаты самой правой текстовой зоны (x1, y1, x2, y2)
        return zone

    # Шаг 3: строим палитру всех цветов и выбираем ближайший к розовому
    def step_3_build_palette_and_main_color(
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Составить палитру цветов и выбрать самый близкий к розовому."""
        # Разворачиваем массив изображения в список RGB-пикселей
        flat = image.reshape(-1, 3)
        # Получаем уникальные цвета — полную палитру изображения
        palette = np.unique(flat, axis=0)
        # Считаем евклидово расстояние каждого цвета палитры до эталонного розового
        distances = np.linalg.norm(palette.astype(np.float32) - PINK_REFERENCE, axis=1)
        # Основной цвет — элемент палитры с минимальным расстоянием до розового
        main_color = palette[int(np.argmin(distances))]

        # Формируем отчёт о палитре и выбранном основном цвете
        result = {
            "palette_size": int(len(palette)),
            "main_color_rgb": main_color.tolist(),
            "distance_to_pink": float(distances.min()),
            "top_pink_candidates": palette[np.argsort(distances)[:5]].tolist(),
        }
        # Печатаем результат шага 3
        print("step_3_build_palette_and_main_color:", json.dumps(result, ensure_ascii=False))
        # Возвращаем полную палитру и выбранный основной цвет
        return palette, main_color

    # Вспомогательная функция: проверяет, совпадают ли пиксели с основным цветом
    def _color_matches(pixels: np.ndarray, main_color: np.ndarray) -> np.ndarray:
        # Вычисляем попиксельное расстояние до основного цвета в RGB
        diff = np.linalg.norm(pixels.astype(np.float32) - main_color.astype(np.float32), axis=-1)
        # Возвращаем булеву маску пикселей в пределах допуска COLOR_TOLERANCE
        return diff <= COLOR_TOLERANCE

    # Вспомогательная функция: переводит бинарную маску SAM2 в ограничивающий прямоугольник
    def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        # Находим координаты всех ненулевых (активных) пикселей маски
        ys, xs = np.where(mask)
        # Если маска пустая — ограничивающий прямоугольник построить нельзя
        if len(xs) == 0:
            return None
        # Возвращаем bbox в формате (x_min, y_min, x_max, y_max)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    # Вспомогательная функция: ищет прямоугольные кандидаты в текстовые области
    def _find_text_candidates(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Найти кандидатов в текстовые области для подсказок SAM2."""
        # Переводим изображение в оттенки серого для бинаризации
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Адаптивная пороговая бинаризация: светлый фон, тёмные символы текста
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )
        # Маленькое прямоугольное ядро для морфологического замыкания разрывов в символах
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Замыкание соединяет близкие фрагменты одного текстового элемента
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Извлекаем внешние контуры связных компонент на бинарной карте
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Список прямоугольников-кандидатов в текстовые области
        candidates: list[tuple[int, int, int, int]] = []
        # Размеры изображения для нормировки порогов фильтрации
        height, width = image.shape[:2]

        # Перебираем каждый найденный контур
        for contour in contours:
            # Строим оси-выровненный ограничивающий прямоугольник контура
            x, y, w, h = cv2.boundingRect(contour)
            # Площадь прямоугольника в пикселях
            area = w * h
            # Отбрасываем слишком мелкие и слишком крупные области
            if area < 30 or area > width * height * 0.08:
                continue
            # Отбрасываем области с неправдоподобной высотой для текста
            if h < 4 or h > height * 0.35:
                continue
            # Отбрасываем слишком узкие области шириной менее 4 пикселей
            if w < 4:
                continue
            # Соотношение сторон ширины к высоте
            aspect = w / max(h, 1)
            # Отбрасываем слишком вытянутые или слишком плоские фигуры
            if aspect > 40 or aspect < 0.15:
                continue
            # Добавляем кандидата в формате (x1, y1, x2, y2)
            candidates.append((x, y, x + w - 1, y + h - 1))

        # Сортируем кандидатов сверху вниз, затем слева направо
        candidates.sort(key=lambda box: (box[1], box[0]))
        # Список объединённых кандидатов после слияния пересекающихся рамок
        merged: list[tuple[int, int, int, int]] = []
        # Проходим по отсортированным кандидатам для объединения соседних рамок
        for box in candidates:
            # Первый кандидат сразу попадает в список объединённых
            if not merged:
                merged.append(box)
                continue
            # Берём последний уже добавленный прямоугольник
            last = merged[-1]
            # Проверяем перекрытие по вертикали с допуском в 2 пикселя
            overlap_y = not (box[3] < last[1] - 2 or box[1] > last[3] + 2)
            # Проверяем перекрытие по горизонтали с допуском в 4 пикселя
            overlap_x = not (box[2] < last[0] - 4 or box[0] > last[2] + 4)
            # Если рамки пересекаются — расширяем последнюю до общего охвата
            if overlap_y and overlap_x:
                merged[-1] = (
                    min(last[0], box[0]),
                    min(last[1], box[1]),
                    max(last[2], box[2]),
                    max(last[3], box[3]),
                )
            else:
                # Иначе добавляем как отдельный независимый кандидат
                merged.append(box)
        # Возвращаем итоговый список кандидатов для подсказок SAM2
        return merged

    # Вспомогательная функция: сегментация SAM2 по прямоугольной подсказке (bounding box)
    def _sam2_segment_box(
        pil_image: Image.Image,
        processor: Sam2Processor,
        model: Sam2Model,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        # Распаковываем координаты исходного кандидата
        x1, y1, x2, y2 = box
        # Небольшой отступ вокруг рамки для захвата краёв символов
        pad = 2
        # Формируем тензор входных рамок в формате, ожидаемом SAM2Processor
        input_boxes = [[[max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad]]]
        # Преобразуем изображение и рамки в тензоры PyTorch на устройстве модели
        inputs = processor(images=pil_image, input_boxes=input_boxes, return_tensors="pt").to(model.device)

        # Отключаем расчёт градиентов — используем модель только для инференса
        with torch.no_grad():
            # Прямой проход модели SAM2 по входным данным
            outputs = model(**inputs)

        # Постобработка масок: приведение к исходному размеру изображения
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        # Оценки IoU для каждой из трёх альтернативных масок SAM2
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        # Выбираем маску с наивысшей уверенностью модели
        best_idx = int(np.argmax(scores))
        # Преобразуем лучшую маску в булев массив
        mask = masks[0, best_idx].numpy().astype(bool)
        # Возвращаем ограничивающий прямоугольник найденной маски
        return _mask_to_bbox(mask)

    # Вспомогательная функция: сегментация SAM2 по точечной подсказке (point prompt)
    def _sam2_segment_point(
        pil_image: Image.Image,
        processor: Sam2Processor,
        model: Sam2Model,
        point: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        # Координаты точки-подсказки внутри предполагаемого текстового объекта
        cx, cy = point
        # Формат точки: одно изображение, один объект, одна точка, координаты (x, y)
        input_points = [[[[cx, cy]]]]
        # Метка 1 означает «передний план» — объект, который нужно выделить
        input_labels = [[[1]]]
        # Подготавливаем входные тензоры с точечной подсказкой
        inputs = processor(
            images=pil_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(model.device)

        # Инференс без вычисления градиентов
        with torch.no_grad():
            outputs = model(**inputs)

        # Приводим предсказанные маски к размеру исходного изображения
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        # Извлекаем оценки качества для каждой альтернативной маски
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        # Индекс маски с максимальным IoU
        best_idx = int(np.argmax(scores))
        # Лучшая маска в виде булева массива
        mask = masks[0, best_idx].numpy().astype(bool)
        # Возвращаем bbox сегментированного объекта
        return _mask_to_bbox(mask)

    # Вспомогательная функция: проверяет, похож ли bbox на текстовую область
    def _is_text_like_bbox(
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int, int],
    ) -> bool:
        # Распаковываем координаты проверяемого прямоугольника
        bx1, by1, bx2, by2 = bbox
        # Ширина ограничивающего прямоугольника в пикселях
        box_w = bx2 - bx1 + 1
        # Высота ограничивающего прямоугольника в пикселях
        box_h = by2 - by1 + 1
        # Отсеиваем слишком маленькие и слишком высокие области
        if box_h < 4 or box_w < 4 or box_h > image_shape[0] * 0.4:
            return False
        # Отсеиваем слишком широкие горизонтальные полосы, не похожие на строки текста
        if box_w / max(box_h, 1) > 35:
            return False
        # Площадь прямоугольника
        area = box_w * box_h
        # Отсеиваем области, занимающие более 5 % площади всего изображения
        if area > image_shape[0] * image_shape[1] * 0.05:
            return False
        # Прямоугольник прошёл все эвристики текстовой области
        return True

    # Шаг 4: детекция всех текстовых областей с помощью модели SAM2
    def step_4_detect_texts_with_sam2(
        image: np.ndarray,
        pil_image: Image.Image,
        rightmost_zone: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        """Найти все тексты с помощью SAM2."""
        # Выбираем CUDA при наличии GPU, иначе вычисления на CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Идентификатор компактной версии SAM2 для быстрой работы в Colab
        model_id = "facebook/sam2.1-hiera-tiny"
        # Загружаем препроцессор, совместимый с выбранной моделью
        processor = Sam2Processor.from_pretrained(model_id)
        # Загружаем веса модели и переносим на выбранное устройство
        model = Sam2Model.from_pretrained(model_id).to(device)
        # Переводим модель в режим оценки (отключает dropout и batch norm train)
        model.eval()

        # Получаем список кандидатов из классической CV-обработки
        candidates = _find_text_candidates(image)
        # Итоговый список обнаруженных текстовых рамок
        text_boxes: list[tuple[int, int, int, int]] = []
        # Множество уже добавленных рамок для исключения дубликатов
        seen_boxes: set[tuple[int, int, int, int]] = set()

        # Для каждого кандидата уточняем границы текста через SAM2 по bbox-подсказке
        for box in candidates:
            # Сегментируем область кандидата моделью SAM2
            bbox = _sam2_segment_box(pil_image, processor, model, box)
            # Пропускаем пустые или неподходящие по размеру результаты
            if bbox is None or not _is_text_like_bbox(bbox, image.shape):
                continue
            # Пропускаем уже учтённые дубликаты
            if bbox in seen_boxes:
                continue
            # Запоминаем рамку как обработанную
            seen_boxes.add(bbox)
            # Добавляем уточнённую рамку в список найденных текстов
            text_boxes.append(bbox)

        # Распаковываем координаты самой правой текстовой зоны
        zone_x1, _, zone_x2, zone_y2 = rightmost_zone
        # Дополнительно сканируем правую зону точечными подсказками по вертикали
        for y in range(8, zone_y2, 14):
            # Центр правой зоны по горизонтали — типичное расположение подписей легенды
            cx = (zone_x1 + zone_x2) // 2
            # Сегментируем объект в точке (cx, y) с помощью SAM2
            bbox = _sam2_segment_point(pil_image, processor, model, (cx, y))
            # Пропускаем неудачные или неподходящие сегментации
            if bbox is None or not _is_text_like_bbox(bbox, image.shape):
                continue
            # Пропускаем дубликаты
            if bbox in seen_boxes:
                continue
            # Регистрируем новую уникальную рамку
            seen_boxes.add(bbox)
            text_boxes.append(bbox)

        # Сортируем найденные рамки для стабильного порядка вывода
        text_boxes.sort(key=lambda box: (box[1], box[0]))
        # Формируем отчёт о детекции текстов
        result = {
            "device": device,
            "sam2_model": model_id,
            "candidate_prompts": len(candidates),
            "detected_text_boxes": [list(box) for box in text_boxes],
            "detected_count": len(text_boxes),
        }
        # Печатаем результат шага 4
        print("step_4_detect_texts_with_sam2:", json.dumps(result, ensure_ascii=False))
        # Возвращаем список всех обнаруженных текстовых рамок
        return text_boxes

    # Шаг 5: фильтрация текстов — оставляем только те, где на горизонтали есть основной цвет
    def step_5_filter_texts_by_main_color(
        image: np.ndarray,
        text_boxes: list[tuple[int, int, int, int]],
        main_color: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Отфильтровать тексты без основного цвета на горизонтальных линиях."""
        # Список рамок, прошедших фильтрацию по основному цвету
        filtered: list[tuple[int, int, int, int]] = []

        # Проверяем каждую обнаруженную текстовую рамку
        for x1, y1, x2, y2 in text_boxes:
            # Флаг: найдена ли хотя бы одна горизонтальная линия с основным цветом
            has_main_color_line = False
            # Перебираем все горизонтальные линии, проходящие через рамку
            for y in range(y1, y2 + 1):
                # Берём все пиксели строки y на всю ширину изображения
                row_pixels = image[y, :, :]
                # Проверяем наличие хотя бы одного пикселя основного цвета в строке
                if np.any(_color_matches(row_pixels, main_color)):
                    has_main_color_line = True
                    break
            # Оставляем рамку, если хотя бы одна горизонталь содержит основной цвет
            if has_main_color_line:
                filtered.append((x1, y1, x2, y2))

        # Формируем отчёт о фильтрации
        result = {
            "input_boxes": len(text_boxes),
            "filtered_boxes": [list(box) for box in filtered],
            "filtered_count": len(filtered),
            "removed_count": len(text_boxes) - len(filtered),
        }
        # Печатаем результат шага 5
        print("step_5_filter_texts_by_main_color:", json.dumps(result, ensure_ascii=False))
        # Возвращаем отфильтрованный список текстовых рамок
        return filtered

    # Шаг 6: замена цветов вне выделений и сохранение результатов в каталог result
    def step_6_save_results(
        image: np.ndarray,
        filtered_boxes: list[tuple[int, int, int, int]],
        main_color: np.ndarray,
    ) -> dict:
        """Заменить цвета вне выделений и сохранить результат."""
        # Создаём каталог result, если он ещё не существует
        result_dir.mkdir(parents=True, exist_ok=True)
        # Копия исходного изображения для формирования итогового результата
        output = image.copy()
        # Булева маска пикселей, попадающих внутрь отфильтрованных текстовых рамок
        text_mask = np.zeros(image.shape[:2], dtype=bool)

        # Отмечаем в маске все пиксели внутри каждой отфильтрованной рамки
        for x1, y1, x2, y2 in filtered_boxes:
            text_mask[y1 : y2 + 1, x1 : x2 + 1] = True

        # Маска пикселей, совпадающих с основным (розовым) цветом
        main_color_mask = _color_matches(output, main_color)
        # Маска области вне текстовых выделений
        outside_text = ~text_mask
        # Заменяем только те пиксели вне текста, которые не совпадают с основным цветом
        replace_mask = outside_text & ~main_color_mask
        # Устанавливаем отмеченные пиксели в белый цвет (255 для всех каналов RGB)
        output[replace_mask] = 255

        # Путь для сохранения обработанного изображения
        output_image_path = result_dir / "output.png"
        # Путь для сохранения координат рамок в формате JSON
        boxes_json_path = result_dir / "filtered_boxes.json"
        # Путь для сохранения визуализации рамок поверх результата
        boxes_vis_path = result_dir / "filtered_boxes.png"

        # Сохраняем итоговое изображение с заменёнными цветами
        Image.fromarray(output).save(output_image_path)

        # Структура данных для JSON-файла с рамками
        boxes_payload = {
            "main_color_rgb": main_color.tolist(),
            "filtered_boxes": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in filtered_boxes
            ],
        }
        # Записываем JSON с рамками на диск в кодировке UTF-8
        boxes_json_path.write_text(
            json.dumps(boxes_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Копия результата для отрисовки зелёных рамок поверх
        vis = output.copy()
        # Рисуем прямоугольник вокруг каждой отфильтрованной текстовой области
        for x1, y1, x2, y2 in filtered_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 1)
        # Сохраняем визуализацию с рамками
        Image.fromarray(vis).save(boxes_vis_path)

        # Формируем отчёт о сохранённых артефактах
        result = {
            "output_image": str(output_image_path),
            "filtered_boxes_json": str(boxes_json_path),
            "filtered_boxes_visualization": str(boxes_vis_path),
            "saved_boxes": len(filtered_boxes),
            "replaced_pixels": int(replace_mask.sum()),
        }
        # Печатаем результат шага 6
        print("step_6_save_results:", json.dumps(result, ensure_ascii=False))
        # Возвращаем словарь с путями и статистикой сохранения
        return result

    # Выполняем шаг 1: загрузка изображения
    image, pil_image = step_1_load_image()
    # Выполняем шаг 2: определение самой правой текстовой зоны
    rightmost_zone = step_2_find_rightmost_text_zone(image)
    # Выполняем шаг 3: построение палитры и выбор основного розового цвета
    palette, main_color = step_3_build_palette_and_main_color(image)
    # Выполняем шаг 4: детекция текстов с помощью SAM2
    text_boxes = step_4_detect_texts_with_sam2(image, pil_image, rightmost_zone)
    # Выполняем шаг 5: фильтрация текстов по наличию основного цвета на горизонталях
    filtered_boxes = step_5_filter_texts_by_main_color(image, text_boxes, main_color)
    # Выполняем шаг 6: замена цветов и сохранение результатов
    save_info = step_6_save_results(image, filtered_boxes, main_color)

    # Итоговый сводный отчёт по всему пайплайну
    final_result = {
        "rightmost_text_zone": list(rightmost_zone),
        "palette_size": int(len(palette)),
        "main_color_rgb": main_color.tolist(),
        "detected_texts": len(text_boxes),
        "filtered_texts": len(filtered_boxes),
        "artifacts": save_info,
    }
    # Печатаем финальный результат работы main()
    print("main:", json.dumps(final_result, ensure_ascii=False))


# Точка входа при запуске скрипта напрямую: python colab_b1/script.py
if __name__ == "__main__":
    main()
