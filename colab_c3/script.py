"""Полный перебор NxN масок для test_c3.jpg."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

# Размер ячейки сетки (определён анализом розовых квадратов на test_c3.jpg:
# наибольшая доля однородных NxN-блоков и шаг смены цвета ≈ 2 пикселя).
N = 2

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR.parent / "test_c3.jpg"
RESULT_DIR = SCRIPT_DIR / "result"


def main() -> None:
    def step_1_load_image() -> Image.Image:
        image = Image.open(IMAGE_PATH).convert("RGB")
        width, height = image.size
        result = {
            "path": str(IMAGE_PATH),
            "width": width,
            "height": height,
            "mode": image.mode,
        }
        print(f"step_1: {result}")
        return image

    def step_2_verify_grid_size(image: Image.Image) -> int:
        pixels = image.load()
        width, height = image.size
        best_n = N
        best_ratio = 0.0

        for candidate in range(2, 7):
            uniform = 0
            total = 0
            for y in range(0, height - candidate + 1, candidate):
                for x in range(0, width - candidate + 1, candidate):
                    total += 1
                    base = pixels[x, y]
                    if all(
                        sum(abs(pixels[x + dx, y + dy][channel] - base[channel]) for channel in range(3))
                        <= 15
                        for dy in range(candidate)
                        for dx in range(candidate)
                    ):
                        uniform += 1
            ratio = uniform / total if total else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_n = candidate

        result = {
            "configured_n": N,
            "detected_n": best_n,
            "uniform_block_ratio": round(best_ratio, 4),
            "match": N == best_n,
        }
        print(f"step_2: {result}")
        return N

    def step_3_prepare_result_dir() -> Path:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        result = {
            "result_dir": str(RESULT_DIR),
            "exists": RESULT_DIR.is_dir(),
        }
        print(f"step_3: {result}")
        return RESULT_DIR

    def step_4_enumerate_masks(
        image: Image.Image,
        square_size: int,
        output_dir: Path,
    ) -> dict[str, int | str]:
        width, height = image.size
        positions_x = width - square_size + 1
        positions_y = height - square_size + 1
        total_masks = positions_x * positions_y
        saved = 0

        for top in range(positions_y):
            for left in range(positions_x):
                mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle(
                    (left, top, left + square_size - 1, top + square_size - 1),
                    fill=255,
                )
                filename = f"mask_x{left:04d}_y{top:04d}.png"
                mask.save(output_dir / filename)
                saved += 1

        result = {
            "square_size": square_size,
            "positions_horizontal": positions_x,
            "positions_vertical": positions_y,
            "total_masks_saved": saved,
            "expected_masks": total_masks,
            "output_dir": str(output_dir),
        }
        print(f"step_4: {result}")
        return result

    image = step_1_load_image()
    square_size = step_2_verify_grid_size(image)
    output_dir = step_3_prepare_result_dir()
    step_4_enumerate_masks(image, square_size, output_dir)


if __name__ == "__main__":
    main()
