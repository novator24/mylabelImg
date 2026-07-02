#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Получение полигонов по цвету из test_b3.jpg."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    image_path = project_dir / "test_b3.jpg"
    result_dir = base_dir / "result"
    n_colors = 32

    def step_1_load_image() -> tuple[np.ndarray, tuple[int, int]]:
        """Загрузка изображения в память."""
        if not image_path.is_file():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        image_rgb = np.array(pil_image, dtype=np.uint8)
        height, width = image_rgb.shape[:2]

        result = {
            "path": str(image_path),
            "shape": [height, width],
            "dtype": str(image_rgb.dtype),
            "channels": image_rgb.shape[2],
        }
        print("step_1_load_image:", json.dumps(result, ensure_ascii=False))
        return image_rgb, (width, height)

    def step_2_quantize_palette(
        image_rgb: np.ndarray,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Упрощение палитры до 32 основных цветов (K-means)."""
        height, width, _ = image_rgb.shape
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels,
            n_colors,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
        quantized = centers_u8[labels.flatten()].reshape(height, width, 3)
        palette = centers_u8.tolist()

        unique_colors = len({tuple(color) for color in palette})
        result = {
            "target_colors": n_colors,
            "palette_size": len(palette),
            "unique_colors": unique_colors,
            "palette_sample": palette[:5],
        }
        print("step_2_quantize_palette:", json.dumps(result, ensure_ascii=False))
        return quantized, palette

    def step_3_split_regions_by_holes(
        quantized: np.ndarray,
    ) -> dict[tuple[int, int, int], list[np.ndarray]]:
        """Для каждого цвета: разрез областей по центрам внутренних дырок."""
        height, width, _ = quantized.shape
        color_masks: dict[tuple[int, int, int], list[np.ndarray]] = {}

        flat_colors = quantized.reshape(-1, 3)
        unique_colors = np.unique(flat_colors, axis=0)

        for color_arr in unique_colors:
            color = tuple(int(v) for v in color_arr)
            mask = np.all(quantized == color_arr, axis=2).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            component_masks: list[np.ndarray] = []

            for label_id in range(1, num_labels):
                component = (labels == label_id).astype(np.uint8)
                split_mask = _split_component_by_hole_centers(component)
                sub_labels_count, sub_labels = cv2.connectedComponents(
                    split_mask, connectivity=8
                )
                for sub_id in range(1, sub_labels_count):
                    component_masks.append((sub_labels == sub_id).astype(np.uint8))

            if component_masks:
                color_masks[color] = component_masks

        total_regions = sum(len(masks) for masks in color_masks.values())
        result = {
            "colors_processed": len(color_masks),
            "total_regions_after_split": total_regions,
            "colors_preview": [
                {"color": list(color), "regions": len(masks)}
                for color, masks in list(color_masks.items())[:5]
            ],
        }
        print("step_3_split_regions_by_holes:", json.dumps(result, ensure_ascii=False))
        return color_masks

    def step_4_extract_polygons(
        color_masks: dict[tuple[int, int, int], list[np.ndarray]],
    ) -> list[dict]:
        """Описание областей списками вершин полигонов."""
        polygons_data: list[dict] = []
        polygon_id = 0

        for color, masks in color_masks.items():
            for region_idx, mask in enumerate(masks):
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if cv2.contourArea(contour) < 4:
                        continue

                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    vertices = approx.reshape(-1, 2).tolist()

                    if len(vertices) < 3:
                        continue

                    polygons_data.append(
                        {
                            "id": polygon_id,
                            "color_rgb": list(color),
                            "region_index": region_idx,
                            "vertex_count": len(vertices),
                            "vertices": vertices,
                        }
                    )
                    polygon_id += 1

        result = {
            "polygon_count": len(polygons_data),
            "preview": [
                {
                    "id": p["id"],
                    "color_rgb": p["color_rgb"],
                    "vertex_count": p["vertex_count"],
                    "first_vertices": p["vertices"][:3],
                }
                for p in polygons_data[:3]
            ],
        }
        print("step_4_extract_polygons:", json.dumps(result, ensure_ascii=False))
        return polygons_data

    def step_5_save_results(
        polygons_data: list[dict],
        palette: list[list[int]],
        image_size: tuple[int, int],
    ) -> Path:
        """Сохранение результата в каталог result."""
        result_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "source_image": str(image_path),
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "palette_colors": n_colors,
            "palette": palette,
            "polygon_count": len(polygons_data),
            "polygons": polygons_data,
        }

        json_path = result_dir / "polygons.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        summary_path = result_dir / "summary.txt"
        with summary_path.open("w", encoding="utf-8") as file:
            file.write(f"Источник: {image_path.name}\n")
            file.write(f"Размер: {image_size[0]}x{image_size[1]}\n")
            file.write(f"Цветов в палитре: {n_colors}\n")
            file.write(f"Полигонов: {len(polygons_data)}\n")

        result = {
            "result_dir": str(result_dir),
            "json_file": str(json_path),
            "summary_file": str(summary_path),
            "polygon_count": len(polygons_data),
        }
        print("step_5_save_results:", json.dumps(result, ensure_ascii=False))
        return json_path

    def _split_component_by_hole_centers(component: np.ndarray) -> np.ndarray:
        """Разделяет компоненту вертикальными/горизонтальными линиями через центры дырок."""
        split_mask = component.copy()
        contours, hierarchy = cv2.findContours(
            component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            return split_mask

        hierarchy = hierarchy[0]
        rows, cols = np.where(component)
        if rows.size == 0:
            return split_mask

        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(cols.min()), int(cols.max())

        for idx, contour in enumerate(contours):
            # Внутренний контур (дырка) имеет родителя и не имеет детей на этом уровне.
            parent = hierarchy[idx][3]
            if parent == -1:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            split_mask[min_row : max_row + 1, cx] = 0
            split_mask[cy, min_col : max_col + 1] = 0

        return split_mask

    image_rgb, image_size = step_1_load_image()
    quantized, palette = step_2_quantize_palette(image_rgb)
    color_masks = step_3_split_regions_by_holes(quantized)
    polygons_data = step_4_extract_polygons(color_masks)
    step_5_save_results(polygons_data, palette, image_size)


if __name__ == "__main__":
    main()
