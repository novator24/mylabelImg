# Указывает интерпретатор Python 3 для запуска скрипта из командной строки
#!/usr/bin/env python3
# Объявляет кодировку исходного файла UTF-8 для корректной работы с кириллицей
# -*- coding: utf-8 -*-
# Краткое описание назначения модуля: извлечение полигонов по цвету
"""Получение полигонов по цвету из test_b3.jpg."""

# Включает отложенную оценку аннотаций типов для совместимости с вложенными функциями
from __future__ import annotations

# Импорт модуля для сериализации результатов в формат JSON
import json
# Импорт класса Path для работы с путями к файлам и каталогам
from pathlib import Path

# Импорт OpenCV для кластеризации, контуров и морфологических операций
import cv2
# Импорт NumPy для работы с многомерными массивами пикселей
import numpy as np
# Импорт Pillow для загрузки исходного изображения
from PIL import Image


# Главная точка входа пайплайна обработки изображения
def main() -> None:
    # Каталог, в котором расположен текущий скрипт
    base_dir = Path(__file__).resolve().parent
    # Корневой каталог проекта (родитель colab_b3)
    project_dir = base_dir.parent
    # Полный путь к исходному изображению test_b3.jpg
    image_path = project_dir / "test_b3.jpg"
    # Каталог для сохранения итоговых файлов результата
    result_dir = base_dir / "result"
    # Целевое число цветов палитры после квантования
    n_colors = 32

    # Шаг 1: загрузка изображения в оперативную память
    def step_1_load_image() -> tuple[np.ndarray, tuple[int, int]]:
        # Документация вложенной функции шага загрузки
        """Загрузка изображения в память."""
        # Проверка существования файла изображения на диске
        if not image_path.is_file():
            # Исключение, если исходный файл не найден
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        # Открытие изображения и приведение к цветовому пространству RGB
        pil_image = Image.open(image_path).convert("RGB")
        # Преобразование PIL-изображения в массив NumPy uint8
        image_rgb = np.array(pil_image, dtype=np.uint8)
        # Извлечение высоты и ширины загруженного изображения
        height, width = image_rgb.shape[:2]

        # Формирование словаря с метаданными загруженного изображения
        result = {
            # Абсолютный путь к исходному файлу
            "path": str(image_path),
            # Размеры изображения в формате [высота, ширина]
            "shape": [height, width],
            # Строковое представление типа данных массива
            "dtype": str(image_rgb.dtype),
            # Число цветовых каналов (R, G, B)
            "channels": image_rgb.shape[2],
        }
        # Вывод результата шага 1 в консоль в формате JSON
        print("step_1_load_image:", json.dumps(result, ensure_ascii=False))
        # Возврат массива пикселей и размеров (ширина, высота)
        return image_rgb, (width, height)

    # Шаг 2: упрощение палитры изображения методом K-means
    def step_2_quantize_palette(
        # Входной RGB-массив исходного изображения
        image_rgb: np.ndarray,
    # Возвращает квантованное изображение и список цветов палитры
    ) -> tuple[np.ndarray, list[list[int]]]:
        # Документация вложенной функции квантования палитры
        """Упрощение палитры до 32 основных цветов (K-means)."""
        # Распаковка размеров изображения и числа каналов
        height, width, _ = image_rgb.shape
        # Преобразование пикселей в матрицу N×3 типа float32 для K-means
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)

        # Критерии остановки алгоритма K-means: точность, итерации, epsilon
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # Запуск K-means: метки кластеров и центры цветов
        _, labels, centers = cv2.kmeans(
            # Входные пиксели для кластеризации
            pixels,
            # Число кластеров (цветов палитры)
            n_colors,
            # Начальные метки (не используются, передаётся None)
            None,
            # Критерии сходимости алгоритма
            criteria,
            # Число повторных запусков с разной инициализацией
            3,
            # Метод инициализации центров: K-means++
            cv2.KMEANS_PP_CENTERS,
        )

        # Приведение центров кластеров к диапазону 0–255 и типу uint8
        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
        # Построение квантованного изображения по меткам кластеров
        quantized = centers_u8[labels.flatten()].reshape(height, width, 3)
        # Преобразование палитры в список списков для сериализации
        palette = centers_u8.tolist()

        # Подсчёт уникальных цветов в палитре (должно быть ≤ n_colors)
        unique_colors = len({tuple(color) for color in palette})
        # Формирование словаря с итогами квантования палитры
        result = {
            # Заданное целевое число цветов
            "target_colors": n_colors,
            # Фактический размер списка палитры
            "palette_size": len(palette),
            # Число уникальных цветов среди центров кластеров
            "unique_colors": unique_colors,
            # Первые 5 цветов палитры для предпросмотра
            "palette_sample": palette[:5],
        }
        # Вывод результата шага 2 в консоль в формате JSON
        print("step_2_quantize_palette:", json.dumps(result, ensure_ascii=False))
        # Возврат квантованного изображения и полной палитры
        return quantized, palette

    # Шаг 3: разделение одноцветных областей по центрам внутренних дырок
    def step_3_split_regions_by_holes(
        # Квантованное RGB-изображение с ограниченной палитрой
        quantized: np.ndarray,
    # Словарь: цвет RGB → список бинарных масок подобластей
    ) -> dict[tuple[int, int, int], list[np.ndarray]]:
        # Документация вложенной функции разрезания областей
        """Для каждого цвета: разрез областей по центрам внутренних дырок."""
        # Распаковка размеров квантованного изображения
        height, width, _ = quantized.shape
        # Итоговый словарь масок для каждого уникального цвета
        color_masks: dict[tuple[int, int, int], list[np.ndarray]] = {}

        # Преобразование изображения в плоский список RGB-пикселей
        flat_colors = quantized.reshape(-1, 3)
        # Получение всех уникальных цветов, присутствующих на изображении
        unique_colors = np.unique(flat_colors, axis=0)

        # Перебор каждого уникального цвета палитры
        for color_arr in unique_colors:
            # Преобразование массива цвета в кортеж целых (R, G, B)
            color = tuple(int(v) for v in color_arr)
            # Бинарная маска: 1 там, где пиксель совпадает с текущим цветом
            mask = np.all(quantized == color_arr, axis=2).astype(np.uint8)

            # Поиск связных компонент данного цвета (8-связность)
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            # Список масок подобластей после разрезания по дыркам
            component_masks: list[np.ndarray] = []

            # Обход каждой связной компоненты, пропуская фон (метка 0)
            for label_id in range(1, num_labels):
                # Бинарная маска одной связной компоненты текущего цвета
                component = (labels == label_id).astype(np.uint8)
                # Разрез компоненты линиями через центры внутренних дырок
                split_mask = _split_component_by_hole_centers(component)
                # Повторный поиск связных компонент после разрезания
                sub_labels_count, sub_labels = cv2.connectedComponents(
                    # Маска компоненты после нанесения разрезающих линий
                    split_mask, connectivity=8
                )
                # Сбор всех новых подобластей, полученных после разреза
                for sub_id in range(1, sub_labels_count):
                    # Добавление маски очередной подобласти в список
                    component_masks.append((sub_labels == sub_id).astype(np.uint8))

            # Сохранение масок только если для цвета найдены области
            if component_masks:
                # Запись списка масок в словарь по ключу цвета
                color_masks[color] = component_masks

        # Общее число областей всех цветов после разрезания
        total_regions = sum(len(masks) for masks in color_masks.values())
        # Формирование словаря с итогами шага разрезания
        result = {
            # Число обработанных уникальных цветов
            "colors_processed": len(color_masks),
            # Суммарное число областей после разреза по дыркам
            "total_regions_after_split": total_regions,
            # Предпросмотр первых 5 цветов и числа их областей
            "colors_preview": [
                # Словарь с цветом и количеством областей для каждого цвета
                {"color": list(color), "regions": len(masks)}
                # Ограничение предпросмотра первыми пятью цветами
                for color, masks in list(color_masks.items())[:5]
            ],
        }
        # Вывод результата шага 3 в консоль в формате JSON
        print("step_3_split_regions_by_holes:", json.dumps(result, ensure_ascii=False))
        # Возврат словаря масок по цветам
        return color_masks

    # Шаг 4: извлечение полигонов как списков вершин из бинарных масок
    def step_4_extract_polygons(
        # Словарь масок областей, сгруппированных по цвету
        color_masks: dict[tuple[int, int, int], list[np.ndarray]],
    # Возвращает список словарей с данными каждого полигона
    ) -> list[dict]:
        # Документация вложенной функции извлечения полигонов
        """Описание областей списками вершин полигонов."""
        # Накопительный список всех найденных полигонов
        polygons_data: list[dict] = []
        # Счётчик уникальных идентификаторов полигонов
        polygon_id = 0

        # Перебор всех цветов и их масок областей
        for color, masks in color_masks.items():
            # Перебор масок областей одного цвета с индексом региона
            for region_idx, mask in enumerate(masks):
                # Поиск внешних контуров на бинарной маске области
                contours, _ = cv2.findContours(
                    # Бинарная маска текущей подобласти
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                # Обработка каждого найденного контура
                for contour in contours:
                    # Пропуск слишком мелких контуров (шум, < 4 пикселей)
                    if cv2.contourArea(contour) < 4:
                        # Переход к следующему контуру без обработки
                        continue

                    # Допуск упрощения контура: 0.2% от длины периметра
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    # Упрощение контура до полигона с заданной точностью
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    # Преобразование вершин в список пар [x, y]
                    vertices = approx.reshape(-1, 2).tolist()

                    # Пропуск вырожденных фигур с менее чем тремя вершинами
                    if len(vertices) < 3:
                        # Переход к следующему контуру
                        continue

                    # Добавление записи о полигоне в итоговый список
                    polygons_data.append(
                        # Словарь с полными данными одного полигона
                        {
                            # Уникальный порядковый идентификатор полигона
                            "id": polygon_id,
                            # RGB-цвет области, которой принадлежит полигон
                            "color_rgb": list(color),
                            # Индекс исходной подобласти внутри цвета
                            "region_index": region_idx,
                            # Число вершин упрощённого полигона
                            "vertex_count": len(vertices),
                            # Список координат вершин [[x, y], ...]
                            "vertices": vertices,
                        }
                    )
                    # Увеличение счётчика для следующего полигона
                    polygon_id += 1

        # Формирование словаря с итогами извлечения полигонов
        result = {
            # Общее число извлечённых полигонов
            "polygon_count": len(polygons_data),
            # Предпросмотр первых трёх полигонов для отладки
            "preview": [
                # Краткие данные одного полигона для предпросмотра
                {
                    # Идентификатор полигона
                    "id": p["id"],
                    # RGB-цвет полигона
                    "color_rgb": p["color_rgb"],
                    # Число вершин полигона
                    "vertex_count": p["vertex_count"],
                    # Первые три вершины для быстрого просмотра
                    "first_vertices": p["vertices"][:3],
                }
                # Ограничение предпросмотра первыми тремя полигонами
                for p in polygons_data[:3]
            ],
        }
        # Вывод результата шага 4 в консоль в формате JSON
        print("step_4_extract_polygons:", json.dumps(result, ensure_ascii=False))
        # Возврат полного списка данных полигонов
        return polygons_data

    # Шаг 5: сохранение полигонов и сводки в каталог result
    def step_5_save_results(
        # Список словарей с данными всех полигонов
        polygons_data: list[dict],
        # Палитра из 32 основных цветов
        palette: list[list[int]],
        # Размер исходного изображения (ширина, высота)
        image_size: tuple[int, int],
    # Возвращает путь к сохранённому JSON-файлу
    ) -> Path:
        # Документация вложенной функции сохранения результатов
        """Сохранение результата в каталог result."""
        # Создание каталога result, если он ещё не существует
        result_dir.mkdir(parents=True, exist_ok=True)

        # Формирование полной структуры данных для JSON-файла
        output = {
            # Путь к исходному изображению
            "source_image": str(image_path),
            # Размеры изображения в виде словаря width/height
            "image_size": {"width": image_size[0], "height": image_size[1]},
            # Число цветов в упрощённой палитре
            "palette_colors": n_colors,
            # Полный список RGB-цветов палитры
            "palette": palette,
            # Общее число сохраняемых полигонов
            "polygon_count": len(polygons_data),
            # Массив всех полигонов с вершинами
            "polygons": polygons_data,
        }

        # Путь к основному JSON-файлу с полигонами
        json_path = result_dir / "polygons.json"
        # Запись JSON-файла с поддержкой кириллицы и отступами
        with json_path.open("w", encoding="utf-8") as file:
            # Сериализация словаря output в файл
            json.dump(output, file, ensure_ascii=False, indent=2)

        # Путь к текстовому файлу со сводной информацией
        summary_path = result_dir / "summary.txt"
        # Запись человекочитаемой сводки в текстовый файл
        with summary_path.open("w", encoding="utf-8") as file:
            # Строка с именем исходного файла
            file.write(f"Источник: {image_path.name}\n")
            # Строка с размерами изображения
            file.write(f"Размер: {image_size[0]}x{image_size[1]}\n")
            # Строка с числом цветов палитры
            file.write(f"Цветов в палитре: {n_colors}\n")
            # Строка с общим числом полигонов
            file.write(f"Полигонов: {len(polygons_data)}\n")

        # Формирование словаря с путями к сохранённым файлам
        result = {
            # Абсолютный путь к каталогу результатов
            "result_dir": str(result_dir),
            # Путь к JSON-файлу с полигонами
            "json_file": str(json_path),
            # Путь к текстовому файлу сводки
            "summary_file": str(summary_path),
            # Число сохранённых полигонов
            "polygon_count": len(polygons_data),
        }
        # Вывод результата шага 5 в консоль в формате JSON
        print("step_5_save_results:", json.dumps(result, ensure_ascii=False))
        # Возврат пути к основному JSON-файлу результата
        return json_path

    # Вспомогательная функция разрезания компоненты по центрам дырок
    def _split_component_by_hole_centers(component: np.ndarray) -> np.ndarray:
        # Документация вспомогательной функции разрезания по дыркам
        """Разделяет компоненту вертикальными/горизонтальными линиями через центры дырок."""
        # Копия маски компоненты для нанесения разрезающих линий
        split_mask = component.copy()
        # Поиск контуров с иерархией для обнаружения внешних границ и дырок
        contours, hierarchy = cv2.findContours(
            # Бинарная маска связной компоненты
            component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        # Если контуры не найдены — возврат исходной маски без изменений
        if hierarchy is None:
            # Возврат неизменённой копии маски компоненты
            return split_mask

        # Извлечение массива иерархии из обёртки OpenCV (первый уровень)
        hierarchy = hierarchy[0]
        # Координаты всех ненулевых пикселей компоненты
        rows, cols = np.where(component)
        # Если компонента пуста — возврат маски без изменений
        if rows.size == 0:
            # Возврат исходной маски при отсутствии пикселей
            return split_mask

        # Минимальная и максимальная строка (границы по вертикали)
        min_row, max_row = int(rows.min()), int(rows.max())
        # Минимальный и максимальный столбец (границы по горизонтали)
        min_col, max_col = int(cols.min()), int(cols.max())

        # Перебор всех контуров с их индексами в иерархии
        for idx, contour in enumerate(contours):
            # Внутренний контур (дырка) имеет родителя и не имеет детей на этом уровне.
            # Индекс родительского контура в иерархии (3-й элемент)
            parent = hierarchy[idx][3]
            # Пропуск внешних контуров — обрабатываем только дырки (parent != -1)
            if parent == -1:
                # Переход к следующему контуру
                continue

            # Вычисление моментов контура для определения центра масс
            moments = cv2.moments(contour)
            # Пропуск вырожденных контуров с нулевой площадью
            if moments["m00"] == 0:
                # Переход к следующему контуру
                continue

            # X-координата центра дырки (центр масс по горизонтали)
            cx = int(moments["m10"] / moments["m00"])
            # Y-координата центра дырки (центр масс по вертикали)
            cy = int(moments["m01"] / moments["m00"])

            # Вертикальная разрезающая линия через центр дырки по всей высоте компоненты
            split_mask[min_row : max_row + 1, cx] = 0
            # Горизонтальная разрезающая линия через центр дырки по всей ширине компоненты
            split_mask[cy, min_col : max_col + 1] = 0

        # Возврат маски компоненты после нанесения всех разрезающих линий
        return split_mask

    # Вызов шага 1: загрузка изображения
    image_rgb, image_size = step_1_load_image()
    # Вызов шага 2: квантование палитры до 32 цветов
    quantized, palette = step_2_quantize_palette(image_rgb)
    # Вызов шага 3: разрезание областей по центрам дырок
    color_masks = step_3_split_regions_by_holes(quantized)
    # Вызов шага 4: извлечение полигонов из масок
    polygons_data = step_4_extract_polygons(color_masks)
    # Вызов шага 5: сохранение результатов в каталог result
    step_5_save_results(polygons_data, palette, image_size)


# Точка входа при запуске скрипта как самостоятельной программы
if __name__ == "__main__":
    # Запуск главной функции пайплайна
    main()
