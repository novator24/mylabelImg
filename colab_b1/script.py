"""Детекция текста с помощью SAM2 на изображении test_b1.png."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor


PINK_REFERENCE = np.array([255, 192, 203], dtype=np.float32)
COLOR_TOLERANCE = 30
MIN_TEXT_ZONE_WIDTH = 15


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    image_path = root / "test_b1.png"
    result_dir = root / "result"

    def step_1_load_image() -> tuple[np.ndarray, Image.Image]:
        """Загрузить test_b1.png в память."""
        if not image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)
        result = {
            "path": str(image_path),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        }
        print("step_1_load_image:", json.dumps(result, ensure_ascii=False))
        return image, pil_image

    def step_2_find_rightmost_text_zone(image: np.ndarray) -> tuple[int, int, int, int]:
        """По вертикальным линиям найти самую правую зону с текстом."""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        dark = (gray < 120).astype(np.float32)

        column_scores = np.zeros(width, dtype=np.float32)
        for x in range(width):
            column_scores[x] = edges[:, x].mean() + dark[:, x].mean() * 80.0

        kernel = np.ones(7, dtype=np.float32) / 7.0
        smooth_scores = np.convolve(column_scores, kernel, mode="same")
        threshold = smooth_scores.max() * 0.28

        zones: list[tuple[int, int]] = []
        in_zone = False
        start = 0
        for x in range(width):
            if smooth_scores[x] >= threshold and not in_zone:
                start = x
                in_zone = True
            elif smooth_scores[x] < threshold and in_zone:
                if x - start >= MIN_TEXT_ZONE_WIDTH:
                    zones.append((start, x - 1))
                in_zone = False
        if in_zone and width - start >= MIN_TEXT_ZONE_WIDTH:
            zones.append((start, width - 1))

        if not zones:
            zone = (int(width * 0.75), 0, width - 1, height - 1)
        else:
            x_start, x_end = max(zones, key=lambda zone_bounds: zone_bounds[0])
            zone = (x_start, 0, x_end, height - 1)

        result = {
            "all_text_zones": [list(zone_bounds) for zone_bounds in zones],
            "rightmost_text_zone": list(zone),
        }
        print("step_2_find_rightmost_text_zone:", json.dumps(result, ensure_ascii=False))
        return zone

    def step_3_build_palette_and_main_color(
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Составить палитру цветов и выбрать самый близкий к розовому."""
        flat = image.reshape(-1, 3)
        palette = np.unique(flat, axis=0)
        distances = np.linalg.norm(palette.astype(np.float32) - PINK_REFERENCE, axis=1)
        main_color = palette[int(np.argmin(distances))]

        result = {
            "palette_size": int(len(palette)),
            "main_color_rgb": main_color.tolist(),
            "distance_to_pink": float(distances.min()),
            "top_pink_candidates": palette[np.argsort(distances)[:5]].tolist(),
        }
        print("step_3_build_palette_and_main_color:", json.dumps(result, ensure_ascii=False))
        return palette, main_color

    def _color_matches(pixels: np.ndarray, main_color: np.ndarray) -> np.ndarray:
        diff = np.linalg.norm(pixels.astype(np.float32) - main_color.astype(np.float32), axis=-1)
        return diff <= COLOR_TOLERANCE

    def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _find_text_candidates(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Найти кандидатов в текстовые области для подсказок SAM2."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int]] = []
        height, width = image.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 30 or area > width * height * 0.08:
                continue
            if h < 4 or h > height * 0.35:
                continue
            if w < 4:
                continue
            aspect = w / max(h, 1)
            if aspect > 40 or aspect < 0.15:
                continue
            candidates.append((x, y, x + w - 1, y + h - 1))

        candidates.sort(key=lambda box: (box[1], box[0]))
        merged: list[tuple[int, int, int, int]] = []
        for box in candidates:
            if not merged:
                merged.append(box)
                continue
            last = merged[-1]
            overlap_y = not (box[3] < last[1] - 2 or box[1] > last[3] + 2)
            overlap_x = not (box[2] < last[0] - 4 or box[0] > last[2] + 4)
            if overlap_y and overlap_x:
                merged[-1] = (
                    min(last[0], box[0]),
                    min(last[1], box[1]),
                    max(last[2], box[2]),
                    max(last[3], box[3]),
                )
            else:
                merged.append(box)
        return merged

    def _sam2_segment_box(
        pil_image: Image.Image,
        processor: Sam2Processor,
        model: Sam2Model,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        pad = 2
        input_boxes = [[[max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad]]]
        inputs = processor(images=pil_image, input_boxes=input_boxes, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        best_idx = int(np.argmax(scores))
        mask = masks[0, best_idx].numpy().astype(bool)
        return _mask_to_bbox(mask)

    def _sam2_segment_point(
        pil_image: Image.Image,
        processor: Sam2Processor,
        model: Sam2Model,
        point: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        cx, cy = point
        input_points = [[[[cx, cy]]]]
        input_labels = [[[1]]]
        inputs = processor(
            images=pil_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        best_idx = int(np.argmax(scores))
        mask = masks[0, best_idx].numpy().astype(bool)
        return _mask_to_bbox(mask)

    def _is_text_like_bbox(
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int, int],
    ) -> bool:
        bx1, by1, bx2, by2 = bbox
        box_w = bx2 - bx1 + 1
        box_h = by2 - by1 + 1
        if box_h < 4 or box_w < 4 or box_h > image_shape[0] * 0.4:
            return False
        if box_w / max(box_h, 1) > 35:
            return False
        area = box_w * box_h
        if area > image_shape[0] * image_shape[1] * 0.05:
            return False
        return True

    def step_4_detect_texts_with_sam2(
        image: np.ndarray,
        pil_image: Image.Image,
        rightmost_zone: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        """Найти все тексты с помощью SAM2."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "facebook/sam2.1-hiera-tiny"
        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(device)
        model.eval()

        candidates = _find_text_candidates(image)
        text_boxes: list[tuple[int, int, int, int]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()

        for box in candidates:
            bbox = _sam2_segment_box(pil_image, processor, model, box)
            if bbox is None or not _is_text_like_bbox(bbox, image.shape):
                continue
            if bbox in seen_boxes:
                continue
            seen_boxes.add(bbox)
            text_boxes.append(bbox)

        zone_x1, _, zone_x2, zone_y2 = rightmost_zone
        for y in range(8, zone_y2, 14):
            cx = (zone_x1 + zone_x2) // 2
            bbox = _sam2_segment_point(pil_image, processor, model, (cx, y))
            if bbox is None or not _is_text_like_bbox(bbox, image.shape):
                continue
            if bbox in seen_boxes:
                continue
            seen_boxes.add(bbox)
            text_boxes.append(bbox)

        text_boxes.sort(key=lambda box: (box[1], box[0]))
        result = {
            "device": device,
            "sam2_model": model_id,
            "candidate_prompts": len(candidates),
            "detected_text_boxes": [list(box) for box in text_boxes],
            "detected_count": len(text_boxes),
        }
        print("step_4_detect_texts_with_sam2:", json.dumps(result, ensure_ascii=False))
        return text_boxes

    def step_5_filter_texts_by_main_color(
        image: np.ndarray,
        text_boxes: list[tuple[int, int, int, int]],
        main_color: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """Отфильтровать тексты без основного цвета на горизонтальных линиях."""
        filtered: list[tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in text_boxes:
            has_main_color_line = False
            for y in range(y1, y2 + 1):
                row_pixels = image[y, :, :]
                if np.any(_color_matches(row_pixels, main_color)):
                    has_main_color_line = True
                    break
            if has_main_color_line:
                filtered.append((x1, y1, x2, y2))

        result = {
            "input_boxes": len(text_boxes),
            "filtered_boxes": [list(box) for box in filtered],
            "filtered_count": len(filtered),
            "removed_count": len(text_boxes) - len(filtered),
        }
        print("step_5_filter_texts_by_main_color:", json.dumps(result, ensure_ascii=False))
        return filtered

    def step_6_save_results(
        image: np.ndarray,
        filtered_boxes: list[tuple[int, int, int, int]],
        main_color: np.ndarray,
    ) -> dict:
        """Заменить цвета вне выделений и сохранить результат."""
        result_dir.mkdir(parents=True, exist_ok=True)
        output = image.copy()
        text_mask = np.zeros(image.shape[:2], dtype=bool)

        for x1, y1, x2, y2 in filtered_boxes:
            text_mask[y1 : y2 + 1, x1 : x2 + 1] = True

        main_color_mask = _color_matches(output, main_color)
        outside_text = ~text_mask
        replace_mask = outside_text & ~main_color_mask
        output[replace_mask] = 255

        output_image_path = result_dir / "output.png"
        boxes_json_path = result_dir / "filtered_boxes.json"
        boxes_vis_path = result_dir / "filtered_boxes.png"

        Image.fromarray(output).save(output_image_path)

        boxes_payload = {
            "main_color_rgb": main_color.tolist(),
            "filtered_boxes": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in filtered_boxes
            ],
        }
        boxes_json_path.write_text(
            json.dumps(boxes_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        vis = output.copy()
        for x1, y1, x2, y2 in filtered_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 1)
        Image.fromarray(vis).save(boxes_vis_path)

        result = {
            "output_image": str(output_image_path),
            "filtered_boxes_json": str(boxes_json_path),
            "filtered_boxes_visualization": str(boxes_vis_path),
            "saved_boxes": len(filtered_boxes),
            "replaced_pixels": int(replace_mask.sum()),
        }
        print("step_6_save_results:", json.dumps(result, ensure_ascii=False))
        return result

    image, pil_image = step_1_load_image()
    rightmost_zone = step_2_find_rightmost_text_zone(image)
    palette, main_color = step_3_build_palette_and_main_color(image)
    text_boxes = step_4_detect_texts_with_sam2(image, pil_image, rightmost_zone)
    filtered_boxes = step_5_filter_texts_by_main_color(image, text_boxes, main_color)
    save_info = step_6_save_results(image, filtered_boxes, main_color)

    final_result = {
        "rightmost_text_zone": list(rightmost_zone),
        "palette_size": int(len(palette)),
        "main_color_rgb": main_color.tolist(),
        "detected_texts": len(text_boxes),
        "filtered_texts": len(filtered_boxes),
        "artifacts": save_info,
    }
    print("main:", json.dumps(final_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
