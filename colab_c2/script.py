#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сбор равномерной выборки снимков талька через OpenRouter (DeepSeek VL)."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
ALL_DIR = ROOT / "all"
PROMPT_FILE = ROOT / "prompt.txt"
TEST_C2_FILE = ROOT / "test_c2.txt"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
SAMPLE_DIR = OUTPUT_DIR / "sample"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-vl2-small")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
TARGET_PER_CLASS = 100
MAX_ATTEMPTS = int(os.environ.get("MAX_CLASSIFY_ATTEMPTS", "5000"))
REQUEST_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", "120"))
RETRY_COUNT = 3
RETRY_DELAY_SEC = 2.0

CLASSES: list[dict[str, Any]] = [
    {"id": 0, "name": "0_other", "title": "другое / не тальк / не определяется"},
    {"id": 1, "name": "1_talc_flaky", "title": "тальк_чешуйчатый (крупные чешуйки, листоватые агрегаты)"},
    {"id": 2, "name": "2_talc_thin", "title": "тальк_тонкий (мелкочешуйчатый, пластинки <1–2 мкм)"},
    {"id": 3, "name": "3_talc_massive", "title": "тальк_массивный (стеатит, фарфоровидный, без чешуйчатости)"},
    {"id": 4, "name": "4_talc_serpentinic", "title": "тальк с серпентином"},
    {"id": 5, "name": "5_talc_chlorite", "title": "тальк с хлоритом"},
    {"id": 6, "name": "6_talc_magnesite", "title": "тальк с магнезитом/доломитом"},
    {"id": 7, "name": "7_talc_quartz", "title": "тальк с кварцем"},
    {"id": 8, "name": "8_talc_fibrous", "title": "тальк_асбестоподобный (волокнистый)"},
    {"id": 9, "name": "9_talc_altered", "title": "тальк_изменённый (гидротермально/метаморфически)"},
]

CLASS_BY_ID = {c["id"]: c for c in CLASSES}
CLASS_BY_NAME = {c["name"]: c for c in CLASSES}
LEGACY_NAME_MAP = {
    "other": 0,
    "talc_flaky": 1,
    "talc_thin": 2,
    "talc_massive": 3,
    "talc_serpentinic": 4,
    "talc_chlorite": 5,
    "talc_magnesite": 6,
    "talc_quartz": 7,
    "talc_fibrous": 8,
    "talc_weathered": 9,
    "talc_altered": 9,
    "talc_weathered/altered": 9,
}


def main() -> None:
    def step_1_load_config() -> dict[str, Any]:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Установите переменную окружения OPENROUTER_API_KEY")

        if not PROMPT_FILE.is_file():
            raise FileNotFoundError(f"Не найден файл промпта: {PROMPT_FILE}")
        if not TEST_C2_FILE.is_file():
            raise FileNotFoundError(f"Не найден файл критериев: {TEST_C2_FILE}")
        if not ALL_DIR.is_dir():
            raise FileNotFoundError(f"Не найден каталог изображений: {ALL_DIR}")

        config = {
            "api_key": api_key,
            "model": DEFAULT_MODEL,
            "root": ROOT,
            "all_dir": ALL_DIR,
            "output_dir": OUTPUT_DIR,
            "sample_dir": SAMPLE_DIR,
            "manifest_file": MANIFEST_FILE,
            "target_per_class": TARGET_PER_CLASS,
            "seed": int(os.environ.get("RANDOM_SEED", "42")),
        }
        print(json.dumps({"step": 1, "config": {k: str(v) for k, v in config.items() if k != "api_key"}}, ensure_ascii=False, indent=2))
        return config

    def step_2_collect_images(config: dict[str, Any]) -> list[Path]:
        roots = [config["all_dir"] / name for name in ("100", "200", "300")]
        images: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path.resolve())

        images = sorted(set(images))
        random.seed(config["seed"])
        random.shuffle(images)

        result = {"step": 2, "total_images": len(images), "roots": [str(r) for r in roots], "preview": [str(p) for p in images[:5]]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return images

    def step_3_build_prompt(config: dict[str, Any]) -> str:
        template = PROMPT_FILE.read_text(encoding="utf-8")
        criteria_source = TEST_C2_FILE.read_text(encoding="utf-8")

        criteria_lines: list[str] = []
        capture = False
        for line in criteria_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("talc_flaky:"):
                capture = True
            if capture:
                if stripped.startswith("Рекомендации по разметке"):
                    break
                criteria_lines.append(line)

        classes_block = "\n".join(f"{c['id']} — {c['name']}: {c['title']}" for c in CLASSES)
        criteria_block = "\n".join(criteria_lines).strip() or criteria_source[:2500]

        prompt = (
            template.replace("{classes_block}", classes_block).replace("{criteria_block}", criteria_block).strip()
        )

        result = {"step": 3, "prompt_chars": len(prompt), "classes": len(CLASSES), "prompt_preview": prompt[:500]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return prompt

    def _encode_image(image_path: Path) -> tuple[str, str]:
        mime, _ = mimetypes.guess_type(str(image_path))
        if not mime:
            mime = "image/jpeg"
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return mime, data

    def _parse_classification(raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()
        if not text:
            return None

        candidates = [text]
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            candidates.insert(0, fence.group(1))
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace:
            candidates.append(brace.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            class_id = payload.get("class_id")
            class_name = str(payload.get("class_name", "")).strip()

            if class_id is None and class_name:
                if class_name in CLASS_BY_NAME:
                    class_id = CLASS_BY_NAME[class_name]["id"]
                else:
                    normalized = class_name.lower().replace(" ", "_")
                    if normalized in LEGACY_NAME_MAP:
                        class_id = LEGACY_NAME_MAP[normalized]

            try:
                class_id = int(class_id)
            except (TypeError, ValueError):
                continue

            if class_id not in CLASS_BY_ID:
                continue

            return {
                "class_id": class_id,
                "class_name": CLASS_BY_ID[class_id]["name"],
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
                "rationale": str(payload.get("rationale", "")).strip(),
                "raw": raw_text,
            }
        return None

    def _classify_image(config: dict[str, Any], prompt: str, image_path: Path) -> dict[str, Any] | None:
        mime, image_b64 = _encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/colab_c2",
            "X-Title": "colab_c2_talc_sampler",
        }
        body = {
            "model": config["model"],
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    ],
                }
            ],
        }

        last_error = ""
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=body,
                    timeout=REQUEST_TIMEOUT,
                )
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue

                payload = response.json()
                choices = payload.get("choices") or []
                if not choices:
                    last_error = "Пустой ответ choices"
                    time.sleep(RETRY_DELAY_SEC * attempt)
                    continue

                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                    )
                parsed = _parse_classification(str(content))
                if parsed:
                    return parsed
                last_error = f"Не удалось распарсить ответ: {str(content)[:300]}"
            except requests.RequestException as exc:
                last_error = str(exc)

            time.sleep(RETRY_DELAY_SEC * attempt)

        print(json.dumps({"warning": "classification_failed", "image": str(image_path), "error": last_error}, ensure_ascii=False))
        return None

    def step_4_classify_and_sample(
        config: dict[str, Any], images: list[Path], prompt: str
    ) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
        counts = {c["id"]: 0 for c in CLASSES}
        sample: dict[int, list[dict[str, Any]]] = {c["id"]: [] for c in CLASSES}
        log: list[dict[str, Any]] = []
        target = config["target_per_class"]

        def quota_full() -> bool:
            return all(counts[cid] >= target for cid in counts)

        attempts = 0
        for image_path in images:
            if quota_full() or attempts >= MAX_ATTEMPTS:
                break
            attempts += 1

            prediction = _classify_image(config, prompt, image_path)
            if prediction is None:
                log.append({"image": str(image_path), "accepted": False, "reason": "parse_or_api_error"})
                continue

            class_id = prediction["class_id"]
            accepted = counts[class_id] < target
            record = {
                "image": str(image_path),
                "class_id": class_id,
                "class_name": prediction["class_name"],
                "confidence": prediction["confidence"],
                "rationale": prediction["rationale"],
                "accepted": accepted,
            }
            log.append(record)

            if accepted:
                counts[class_id] += 1
                sample[class_id].append(record)

        result = {
            "step": 4,
            "attempts": attempts,
            "counts": counts,
            "total_selected": sum(counts.values()),
            "quota_full": quota_full(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return sample, log

    def step_5_save_sample(
        config: dict[str, Any], sample: dict[int, list[dict[str, Any]]], log: list[dict[str, Any]]
    ) -> Path:
        output_dir: Path = config["output_dir"]
        sample_dir: Path = config["sample_dir"]
        manifest_file: Path = config["manifest_file"]

        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, Any]] = []
        for class_id in sorted(sample.keys()):
            class_name = CLASS_BY_ID[class_id]["name"]
            class_dir = sample_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for idx, item in enumerate(sample[class_id], start=1):
                src = Path(item["image"])
                dst_name = f"{idx:03d}_{src.stem}{src.suffix.lower()}"
                dst = class_dir / dst_name
                shutil.copy2(src, dst)
                row = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "source_image": str(src),
                    "sample_image": str(dst),
                    "confidence": item.get("confidence"),
                    "rationale": item.get("rationale"),
                }
                manifest_rows.append(row)

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model": config["model"],
            "target_per_class": config["target_per_class"],
            "counts": {CLASS_BY_ID[cid]["name"]: len(sample[cid]) for cid in sorted(sample)},
            "items": manifest_rows,
            "classification_log": log,
        }
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        result = {
            "step": 5,
            "manifest": str(manifest_file),
            "sample_dir": str(sample_dir),
            "saved_images": len(manifest_rows),
            "counts": manifest["counts"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return manifest_file

    config = step_1_load_config()
    images = step_2_collect_images(config)
    prompt = step_3_build_prompt(config)
    sample, log = step_4_classify_and_sample(config, images, prompt)
    step_5_save_sample(config, sample, log)


if __name__ == "__main__":
    main()
