"""Сопоставление строк test_b2.txt с вопросом: содержит ли строка элемент Ca."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import requests

MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
NUM_RUNS = 20
TEMPERATURE = 0.3
REQUEST_TIMEOUT = 120


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    input_path = root_dir / "test_b2.txt"
    result_dir = root_dir / "result"

    def step_1_load_lines() -> list[str]:
        """Загрузить test_b2.txt в память."""
        lines = input_path.read_text(encoding="utf-8").splitlines()
        print(f"step_1: загружено {len(lines)} строк из {input_path}")
        return lines

    def step_2_prepare_dirs() -> Path:
        """Подготовить каталог result."""
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"step_2: каталог результатов готов — {result_dir}")
        return result_dir

    def step_3_build_prompt(line_text: str) -> list[dict[str, str]]:
        """Сформировать улучшенный промпт для классификации Ca."""
        system_prompt = (
            "Ты эксперт по минералогии и геохимии. "
            "Твоя единственная задача — определить, относится ли строка к кальцию (химический элемент Ca, кальций).\n\n"
            "Отвечай строго одним словом: YES или NO.\n\n"
            "Правила YES:\n"
            "- в строке явно есть подстрока 'Ca' как символ элемента (например: REE-Ca-Si-P, Fe-Ti-Ca-oxides);\n"
            "- строка называет минерал или группу, где кальций — обязательный или характерный компонент "
            "(например: Calcite, Apatite, Dolomite).\n\n"
            "Правила NO:\n"
            "- строка — только число, единица измерения, техника анализа или общая подпись без Ca "
            "(например: 0.21, area%, 200 um, BSE, Quartz, Feldspars);\n"
            "- в строке нет 'Ca' и нет минерала/группы, для которых кальций типичен.\n\n"
            "Не добавляй пояснений. Формат ответа: YES или NO."
        )
        user_prompt = (
            f"Строка для анализа:\n"
            f"\"{line_text}\"\n\n"
            f"Вопрос: содержит ли эта строка элемент Ca (кальций) — явно или через минерал/группу с Ca?\n"
            f"Ответ (только YES или NO):"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def step_4_parse_answer(raw: str) -> str:
        """Извлечь YES/NO из ответа модели."""
        text = raw.strip().upper()
        if re.search(r"\bYES\b", text):
            return "YES"
        if re.search(r"\bNO\b", text):
            return "NO"
        if "ДА" in text or text == "Y":
            return "YES"
        if "НЕТ" in text or text == "N":
            return "NO"
        return "NO"

    def step_5_query_ollama(messages: list[dict[str, str]]) -> str:
        """Один запрос к локальной модели Ollama через REST API."""
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "options": {"temperature": TEMPERATURE},
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return step_4_parse_answer(content)

    def step_6_majority_vote(line_text: str, line_index: int) -> tuple[str, Counter[str]]:
        """20 запусков подряд; итог — самый частый ответ."""
        messages = step_3_build_prompt(line_text)
        votes: list[str] = []
        for run in range(1, NUM_RUNS + 1):
            answer = step_5_query_ollama(messages)
            votes.append(answer)
            print(f"  строка {line_index:02d}, запуск {run:02d}/{NUM_RUNS}: {answer}")

        counter = Counter(votes)
        winner, count = counter.most_common(1)[0]
        print(
            f"step_6 [строка {line_index:02d}]: \"{line_text}\" -> {winner} "
            f"(голоса: {dict(counter)}, победитель {count}/{NUM_RUNS})"
        )
        return winner, counter

    def step_7_classify_all(lines: list[str]) -> list[tuple[int, str, str, Counter[str]]]:
        """Классифицировать все строки."""
        results: list[tuple[int, str, str, Counter[str]]] = []
        for index, line in enumerate(lines, start=1):
            label, votes = step_6_majority_vote(line, index)
            results.append((index, line, label, votes))
        summary = ", ".join(f"{idx}:{label}" for idx, _, label, _ in results)
        print(f"step_7: классификация завершена — {summary}")
        return results

    def step_8_save_results(
        output_dir: Path,
        results: list[tuple[int, str, str, Counter[str]]],
    ) -> list[Path]:
        """Сохранить YES/NO для каждой строки в каталог result."""
        saved_paths: list[Path] = []

        summary_path = output_dir / "results.txt"
        summary_lines = [label for _, _, label, _ in results]
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        saved_paths.append(summary_path)

        detailed_path = output_dir / "results_detailed.txt"
        detailed_lines = [
            f"{index:02d}\t{label}\t{line}\tvotes={dict(votes)}"
            for index, line, label, votes in results
        ]
        detailed_path.write_text("\n".join(detailed_lines) + "\n", encoding="utf-8")
        saved_paths.append(detailed_path)

        for index, line, label, votes in results:
            line_path = output_dir / f"line_{index:02d}.txt"
            line_path.write_text(
                f"line={index}\n"
                f"text={line}\n"
                f"answer={label}\n"
                f"votes={dict(votes)}\n"
                f"runs={NUM_RUNS}\n",
                encoding="utf-8",
            )
            saved_paths.append(line_path)

        print(f"step_8: сохранено {len(saved_paths)} файлов в {output_dir}")
        return saved_paths

    lines = step_1_load_lines()
    output_dir = step_2_prepare_dirs()
    results = step_7_classify_all(lines)
    saved_paths = step_8_save_results(output_dir, results)

    print("main: готово")
    print(f"  модель: {MODEL_NAME}")
    print(f"  запусков на строку: {NUM_RUNS}")
    print(f"  итоговый файл: {saved_paths[0]}")


if __name__ == "__main__":
    main()
