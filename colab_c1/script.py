# Указывает интерпретатор Python 3 для запуска скрипта из командной строки
#!/usr/bin/env python3
# Многострочная строка документации модуля с описанием назначения скрипта
"""
Развёртывание Ansible и MiMo Code на целевом Ubuntu-стенде через прокси-сервер.

Прокси: вход по логину/паролю.
Целевой хост: вход по SSH-ключам id_rsa/id_rsa.pub с прокси (~/.ssh/).

Документация MiMo Code: https://mimo.xiaomi.com/mimocode/install
"""

# Включает отложенную оценку аннотаций типов для совместимости с Python 3.7+
from __future__ import annotations

# Импорт модуля разбора аргументов командной строки
import argparse
# Импорт модуля доступа к переменным окружения операционной системы
import os
# Импорт модуля системных параметров и потоков ввода-вывода
import sys
# Импорт модуля для удаления общих отступов из многострочных строк
import textwrap
# Импорт класса Path для работы с путями файловой системы
from pathlib import Path
# Импорт типов Any и TYPE_CHECKING для аннотаций и условного импорта
from typing import Any, TYPE_CHECKING

# Импорт библиотеки для чтения и записи YAML-конфигурации
import yaml

# Блок условного импорта только для статической проверки типов (не в runtime)
if TYPE_CHECKING:
    # Импорт paramiko для подсказок типов IDE и mypy
    import paramiko

# Абсолютный путь к каталогу, в котором расположен этот скрипт
SCRIPT_DIR = Path(__file__).resolve().parent
# Путь к основному файлу конфигурации config.yaml по умолчанию
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
# Путь к примеру конфигурации для копирования пользователем
EXAMPLE_CONFIG = SCRIPT_DIR / "config.example.yaml"

# Официальный URL установочного скрипта MiMo Code для Linux
MIMO_INSTALL_URL = "https://mimo.xiaomi.com/install"
# Ожидаемый вывод uname -a на прокси и целевом Ubuntu-стенде
TARGET_UNAME = (
    # Первая часть строки uname -a (ядро и дата сборки)
    "Linux team023 5.15.0-177-generic #187-Ubuntu SMP "
    # Вторая часть строки uname -a (архитектура x86_64 GNU/Linux)
    "Sat Apr 11 22:54:33 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux"
)


# Главная функция скрипта, возвращает код завершения процесса (0 — успех)
def main() -> int:
    # Docstring главной функции: описание точки входа
    """Точка входа: установка или удаление Ansible и MiMo Code."""

    # Вспомогательная функция форматированного вывода результата каждого шага
    def _print_step(title: str, result: Any) -> Any:
        # Печать верхней разделительной линии блока шага
        print(f"\n{'=' * 72}")
        # Печать заголовка текущего шага
        print(f"  {title}")
        # Печать нижней разделительной линии блока шага
        print(f"{'=' * 72}")
        # Если результат — словарь, выводим пары ключ: значение
        if isinstance(result, dict):
            # Перебор всех элементов словаря результата
            for key, value in result.items():
                # Печать одной пары ключ: значение с отступом
                print(f"  {key}: {value}")
        # Если результат — список строк (например, команды Git Bash)
        elif isinstance(result, list):
            # Перебор и печать каждой строки списка
            for line in result:
                print(f"  {line}")
        # Для любого другого типа результата
        else:
            # Печать результата как есть
            print(f"  {result}")
        # Пустая строка после блока вывода для читаемости
        print()
        # Возврат исходного результата вызывающему коду
        return result

    # Шаг 1: разбор аргументов командной строки
    def step_1_parse_args() -> argparse.Namespace:
        # Создание парсера аргументов с описанием назначения программы
        parser = argparse.ArgumentParser(
            # Текст справки, отображаемый в --help
            description="Развёртывание Ansible и MiMo Code через прокси-сервер",
        )
        # Опция -c/--config: путь к файлу конфигурации
        parser.add_argument(
            # Короткое имя опции конфигурации
            "-c",
            # Длинное имя опции конфигурации
            "--config",
            # Значение по умолчанию — DEFAULT_CONFIG
            default=str(DEFAULT_CONFIG),
            # Текст подсказки для опции --config
            help=f"Путь к config.yaml (по умолчанию: {DEFAULT_CONFIG})",
        )
        # Флаг --uninstall: режим удаления ПО с целевого стенда
        parser.add_argument(
            # Имя флага удаления
            "--uninstall",
            # Флаг без значения: присутствует = True
            action="store_true",
            # Текст подсказки для --uninstall
            help="Удалить Ansible и MiMo Code с целевого стенда",
        )
        # Флаг --dry-run: только показать команды без выполнения на серверах
        parser.add_argument(
            # Имя флага пробного запуска
            "--dry-run",
            # Флаг без значения: присутствует = True
            action="store_true",
            # Текст подсказки для --dry-run
            help="Только показать команды, без выполнения на серверах",
        )
        # Флаг --windows-only: только инструкции для Git Bash на Windows
        parser.add_argument(
            # Имя флага только для Windows-инструкций
            "--windows-only",
            # Флаг без значения: присутствует = True
            action="store_true",
            # Текст подсказки для --windows-only
            help="Только вывести команды для Git Bash на Windows",
        )
        # Разбор sys.argv и получение объекта с аргументами
        args = parser.parse_args()
        # Вывод результата шага 1 в консоль
        _print_step("step_1: аргументы CLI", vars(args))
        # Возврат разобранных аргументов
        return args

    # Шаг 2: загрузка и валидация YAML-конфигурации
    def step_2_load_config(config_path: str) -> dict[str, Any]:
        # Преобразование строкового пути в объект Path
        path = Path(config_path)
        # Проверка существования файла конфигурации на диске
        if not path.exists():
            # Формирование подсказки с командой копирования примера конфига
            hint = (
                # Первая строка сообщения об отсутствии config.yaml
                f"Файл {path} не найден. Скопируйте пример:\n"
                # Вторая строка — команда копирования config.example.yaml
                f"  cp {EXAMPLE_CONFIG} {DEFAULT_CONFIG}"
            )
            # Прерывание с понятной ошибкой, если config.yaml отсутствует
            raise FileNotFoundError(hint)

        # Открытие файла конфигурации в кодировке UTF-8
        with path.open(encoding="utf-8") as fh:
            # Парсинг YAML; пустой файл даёт пустой словарь
            cfg = yaml.safe_load(fh) or {}

        # Секция proxy: создаётся пустой словарь, если отсутствует в файле
        proxy = cfg.setdefault("proxy", {})
        # Секция target: хост и пользователь целевого стенда
        target = cfg.setdefault("target", {})
        # Секция mimo: параметры установки MiMo Code
        cfg.setdefault("mimo", {})

        # Если пароль прокси не указан в YAML
        if not proxy.get("password"):
            # Подставляем пароль из переменной окружения PROXY_PASSWORD
            proxy["password"] = os.environ.get("PROXY_PASSWORD", "")

        # Сводка загруженной конфигурации для вывода пользователю
        summary = {
            # Путь к загруженному файлу конфигурации
            "config_file": str(path),
            # Строка подключения к прокси user@host:port
            "proxy": f"{proxy.get('user')}@{proxy.get('host')}:{proxy.get('port', 22)}",
            # Строка подключения к целевому стенду user@host
            "target": f"{target.get('user')}@{target.get('host')}",
            # Статус наличия пароля прокси
            "proxy_auth": "password" if proxy.get("password") else "PROXY_PASSWORD не задан",
            # Путь к SSH-ключу на прокси для доступа к стенду
            "target_key": target.get("private_key", "~/.ssh/id_rsa"),
            # URL установки MiMo Code
            "mimo_url": cfg["mimo"].get("install_url", MIMO_INSTALL_URL),
        }
        # Сохранение сводки внутри cfg для возможного повторного использования
        cfg["_summary"] = summary
        # Печать результата шага 2
        _print_step("step_2: загрузка config.yaml", summary)
        # Возврат полного словаря конфигурации
        return cfg

    # Шаг 3: формирование списка команд для Git Bash на Windows
    def step_3_print_windows_git_bash_commands(cfg: dict[str, Any]) -> list[str]:
        # Docstring: назначение шага — локальная установка на Windows
        """Команды для установки на локальной Windows-машине (Git Bash)."""
        # Извлечение секции целевого хоста из конфигурации
        target = cfg["target"]
        # Извлечение секции прокси-сервера из конфигурации
        proxy = cfg["proxy"]

        # Список команд и комментариев для копирования в Git Bash
        commands = [
            # Заголовок секции подготовки в Git Bash
            "# --- Git Bash на Windows: подготовка ---",
            # Подсказка перейти в каталог проекта
            "# перейдите в каталог colab_c1 вашего репозитория",
            # Переход в рабочий каталог colab_c1
            "cd colab_c1",
            # Установка Python-зависимостей скрипта
            "python -m pip install -r requirements.txt",
            # Проверка версии Python
            "python --version",
            # Проверка версии Node.js (нужен 18+ для MiMo)
            "node --version   # для MiMo Code нужен Node.js 18+",
            # Пустая строка-разделитель в выводе
            "",
            # Заголовок секции установки Ansible
            "# --- Ansible (через pip в Git Bash) ---",
            # Обновление pip перед установкой пакетов
            "python -m pip install --upgrade pip",
            # Установка Ansible через pip
            "python -m pip install ansible",
            # Проверка успешной установки Ansible
            "ansible --version",
            # Пустая строка-разделитель
            "",
            # Заголовок секции установки MiMo Code
            "# --- MiMo Code (официальный способ для Windows) ---",
            # Ссылка на официальную документацию MiMo
            "# https://mimo.xiaomi.com/mimocode/install",
            # Установка MiMo CLI через npm (рекомендуется для Windows)
            "npm install -g @mimo-ai/cli",
            # Подсказка при ошибке выбора платформенного пакета
            "# при ошибке платформы:",
            # Альтернативный npm-пакет для Windows x64
            "# npm install -g @mimo-ai/mimocode-windows-x64",
            # Проверка версии установленного MiMo
            "mimo --version",
            # Пустая строка-разделитель
            "",
            # Заголовок секции настройки конфигурации
            "# --- Конфиг: скопировать и заполнить ---",
            # Копирование примера конфига в рабочий config.yaml
            f"cp config.example.yaml config.yaml",
            # Подсказка задать пароль прокси
            "# задать PROXY_PASSWORD или proxy.password в config.yaml",
            # Пустая строка-разделитель
            "",
            # Заголовок секции запуска скрипта развёртывания
            "# --- Запуск скрипта развёртывания ---",
            # Экспорт пароля прокси в переменную окружения
            "export PROXY_PASSWORD='your_proxy_password'",
            # Пробный запуск без реального SSH
            "python script.py --dry-run",
            # Полный запуск установки на стенд
            "python script.py",
            # Пустая строка-разделитель
            "",
            # Заголовок секции проверки SSH до прокси
            "# --- Проверка SSH до прокси ---",
            # Команда SSH-подключения к прокси-серверу
            f"ssh -p {proxy.get('port', 22)} {proxy['user']}@{proxy['host']}",
            # Пустая строка-разделитель
            "",
            # Заголовок секции проверки с прокси на стенд
            "# --- С прокси: проверка ключей и доступа к стенду ---",
            # Проверка наличия ключей id_rsa на прокси
            "ls -la ~/.ssh/id_rsa ~/.ssh/id_rsa.pub",
            # SSH на целевой стенд с ключом и проверкой uname -a
            (
                # Первая часть команды ssh к целевому хосту
                f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=accept-new "
                # Вторая часть: пользователь, хост и удалённая команда uname -a
                f"{target['user']}@{target['host']} 'uname -a'"
            ),
            # Пустая строка-разделитель
            "",
            # Заголовок секции проверки Ansible ping
            "# --- Ansible ping (после шага 7) ---",
            # Проверка доступности стенда через ansible ping
            f"ansible -i {SCRIPT_DIR / 'inventory.ini'} stand -m ping",
            # Пустая строка-разделитель
            "",
            # Заголовок секции локального удаления на Windows
            "# --- Удаление с Windows (локально) ---",
            # Удаление Ansible, установленного через pip
            "python -m pip uninstall -y ansible",
            # Удаление MiMo CLI, установленного через npm
            "npm uninstall -g @mimo-ai/cli",
            # Пустая строка-разделитель
            "",
            # Заголовок секции удаления с целевого стенда
            "# --- Удаление с целевого Ubuntu-стенда ---",
            # Запуск режима --uninstall скрипта
            "python script.py --uninstall",
            # Подсказка: ручное удаление MiMo на Linux
            "# MiMo на Linux: mimo uninstall --force",
        ]
        # Печать и возврат списка команд шага 3
        return _print_step("step_3: команды для Git Bash (Windows)", commands)

    # Внутренняя функция: выполнение shell-команды на целевом хосте через прокси
    def _ssh_exec_on_target(
        # Полный словарь конфигурации proxy/target/mimo
        cfg: dict[str, Any],
        # Shell-команда для выполнения на целевом хосте
        remote_command: str,
        # Запрет передачи dry_run как позиционного аргумента
        *,
        # True — не подключаться, только вернуть сформированную команду
        dry_run: bool,
    ) -> dict[str, Any]:
        # Данные прокси из конфигурации
        proxy = cfg["proxy"]
        # Данные целевого стенда из конфигурации
        target = cfg["target"]
        # Путь к приватному ключу SSH на прокси (по умолчанию ~/.ssh/id_rsa)
        key = target.get("private_key", "~/.ssh/id_rsa")
        # Порт SSH прокси (по умолчанию 22)
        port = proxy.get("port", 22)

        # Экранирование одинарных кавычек для безопасной передачи в ssh '...'
        escaped = remote_command.replace("'", "'\"'\"'")
        # Команда ssh с прокси на целевой хост (выполняется на прокси)
        via_proxy = (
            # Первая часть ssh: ключ, BatchMode и StrictHostKeyChecking
            f"ssh -i {key} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            # Вторая часть: user@host и экранированная удалённая команда
            f"{target['user']}@{target['host']} '{escaped}'"
        )

        # В режиме dry-run не подключаемся к серверам
        if dry_run:
            # Возврат только сформированной команды для просмотра
            return {
                # Маркер режима без реального выполнения
                "status": "dry-run",
                # Сформированная команда ssh через прокси для ручной проверки
                "via_proxy_command": via_proxy,
            }

        # Ленивый импорт paramiko — только при реальном SSH-подключении
        import paramiko

        # Создание SSH-клиента paramiko
        client = paramiko.SSHClient()
        # Автоматическое принятие неизвестных host key (для автоматизации)
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Подключение к прокси по паролю или ключу
        client.connect(
            # IP или hostname прокси-сервера
            hostname=proxy["host"],
            # Порт SSH прокси
            port=port,
            # Имя пользователя на прокси
            username=proxy["user"],
            # Пароль прокси (None — если используется только ключ/agent)
            password=proxy.get("password") or None,
            # Искать ключи в ~/.ssh, если пароль не задан
            look_for_keys=not proxy.get("password"),
            # Разрешить использование ssh-agent
            allow_agent=True,
            # Таймаут установки соединения в секундах
            timeout=30,
        )
        # Гарантированное закрытие SSH-сессии после выполнения команды
        try:
            # Выполнение команды ssh-to-target на прокси с псевдотерминалом
            _stdin, stdout, stderr = client.exec_command(via_proxy, get_pty=True)
            # Ожидание завершения удалённой команды и получение кода выхода
            exit_code = stdout.channel.recv_exit_status()
            # Чтение стандартного вывода удалённой команды
            out = stdout.read().decode("utf-8", errors="replace").strip()
            # Чтение стандартного потока ошибок удалённой команды
            err = stderr.read().decode("utf-8", errors="replace").strip()
            # Сборка словаря с результатом выполнения
            return {
                # Код завершения удалённой команды (0 = успех)
                "exit_code": exit_code,
                # Стандартный вывод удалённой команды
                "stdout": out,
                # Стандартный поток ошибок удалённой команды
                "stderr": err,
                # Полная команда ssh, выполненная на прокси
                "command": via_proxy,
            }
        # Блок finally выполняется всегда, даже при исключении
        finally:
            # Закрытие SSH-соединения с прокси
            client.close()

    # Шаг 4: проверка цепочки прокси → целевой Ubuntu
    def step_4_verify_connectivity(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        # Запуск диагностической команды на целевом хосте через прокси
        result = _ssh_exec_on_target(
            # Конфигурация с данными прокси и целевого хоста
            cfg,
            # Диагностика: ядро, пользователь и наличие SSH-ключей
            "uname -a && whoami && ls -la ~/.ssh/id_rsa ~/.ssh/id_rsa.pub",
            # Передача флага dry-run в исполнитель SSH
            dry_run=dry_run,
        )
        # При реальном запуске проверяем успешность подключения
        if not dry_run and result.get("exit_code", 1) != 0:
            # Исключение с деталями, если SSH или ключи недоступны
            raise RuntimeError(
                # Текст ошибки с полным результатом неудачного SSH
                f"Не удалось подключиться к целевому стенду через прокси:\n{result}"
            )
        # Печать и возврат результата шага 4
        return _print_step("step_4: проверка прокси → целевой Ubuntu", result)

    # Шаг 5: установка Ansible на целевом Ubuntu-стенде
    def step_5_install_ansible(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        # Bash-скрипт установки Ansible (apt или pip fallback)
        install_script = textwrap.dedent(
            """\
            set -e
            if command -v ansible >/dev/null 2>&1; then
              echo "ANSIBLE_ALREADY=$(ansible --version | head -1)"
              exit 0
            fi
            export DEBIAN_FRONTEND=noninteractive
            if command -v apt-get >/dev/null 2>&1; then
              sudo apt-get update -qq
              sudo apt-get install -y -qq software-properties-common
              sudo apt-get install -y -qq ansible sshpass
            else
              python3 -m pip install --user ansible
              export PATH="$HOME/.local/bin:$PATH"
            fi
            ansible --version | head -1
            """
        ).strip()
        # Выполнение скрипта установки на целевом хосте
        result = _ssh_exec_on_target(cfg, install_script, dry_run=dry_run)
        # Печать и возврат результата шага 5
        return _print_step("step_5: установка Ansible на целевом стенде", result)

    # Шаг 6: установка MiMo Code на целевом Ubuntu-стенде
    def step_6_install_mimo(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        # URL установки MiMo из конфига или константы по умолчанию
        mimo_url = cfg.get("mimo", {}).get("install_url", MIMO_INSTALL_URL)
        # Bash-скрипт официальной установки MiMo через curl | bash
        install_script = textwrap.dedent(
            f"""\
            set -e
            export HOME="${{HOME:-/home/$(whoami)}}"
            if command -v mimo >/dev/null 2>&1; then
              echo "MIMO_ALREADY=$(mimo --version 2>/dev/null || true)"
              exit 0
            fi
            curl -fsSL {mimo_url} | bash
            export PATH="$HOME/.local/bin:$HOME/.mimo/bin:$PATH"
            mimo --version
            """
        ).strip()
        # Выполнение скрипта установки MiMo на целевом хосте
        result = _ssh_exec_on_target(cfg, install_script, dry_run=dry_run)
        # Печать и возврат результата шага 6
        return _print_step("step_6: установка MiMo Code на целевом стенде", result)

    # Шаг 7: генерация inventory.ini и ansible.cfg для Ansible
    def step_7_write_ansible_inventory(cfg: dict[str, Any]) -> dict[str, str]:
        # Секция целевого хоста
        target = cfg["target"]
        # Секция прокси для настройки ProxyJump
        proxy = cfg["proxy"]
        # Путь к SSH-ключу для ansible_ssh_private_key_file
        key = target.get("private_key", "~/.ssh/id_rsa")
        # Строка опции ProxyJump для подключения через прокси с Windows
        proxy_jump = (
            # Опция SSH ProxyJump через прокси user@host:port
            f"-o ProxyJump={proxy['user']}@{proxy['host']}:{proxy.get('port', 22)}"
        )

        # Путь к файлу inventory в каталоге скрипта
        inventory_path = SCRIPT_DIR / "inventory.ini"
        # Содержимое inventory с группой [stand] и переменными
        inventory_content = textwrap.dedent(
            f"""\
            # Сгенерировано script.py
            # Целевой стенд: {TARGET_UNAME}

            [stand]
            {target['host']} ansible_user={target['user']} ansible_ssh_private_key_file={key}

            [stand:vars]
            ansible_python_interpreter=/usr/bin/python3
            ansible_ssh_common_args='-o StrictHostKeyChecking=accept-new {proxy_jump}'
            """
        ).strip() + "\n"
        # Запись inventory.ini на диск в UTF-8
        inventory_path.write_text(inventory_content, encoding="utf-8")

        # Путь к локальному ansible.cfg
        ansible_cfg_path = SCRIPT_DIR / "ansible.cfg"
        # Содержимое ansible.cfg с inventory и параметрами SSH
        ansible_cfg_content = textwrap.dedent(
            f"""\
            [defaults]
            inventory = {inventory_path.name}
            host_key_checking = False
            retry_files_enabled = False
            stdout_callback = yaml

            [ssh_connection]
            pipelining = True
            """
        ).strip() + "\n"
        # Запись ansible.cfg на диск в UTF-8
        ansible_cfg_path.write_text(ansible_cfg_content, encoding="utf-8")

        # Словарь с путями и подсказками для пользователя
        result = {
            # Абсолютный путь к сгенерированному inventory.ini
            "inventory": str(inventory_path),
            # Абсолютный путь к сгенерированному ansible.cfg
            "ansible_cfg": str(ansible_cfg_path),
            # Готовая команда проверки доступности стенда
            "ping_command": f"ansible -i {inventory_path} stand -m ping",
            # Подсказка, откуда запускать ansible
            "note": (
                # Первая часть текста подсказки
                "Запускайте ansible из каталога colab_c1 или укажите "
                # Вторая часть: переменная ANSIBLE_CONFIG с путём к cfg
                f"ANSIBLE_CONFIG={ansible_cfg_path}"
            ),
        }
        # Печать и возврат результата шага 7
        return _print_step("step_7: конфиг Ansible (inventory + ansible.cfg)", result)

    # Шаг 8: удаление Ansible и MiMo Code с целевого стенда
    def step_8_uninstall(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        # Bash-скрипт полного удаления MiMo и Ansible
        uninstall_script = textwrap.dedent(
            """\
            set -e
            echo "=== Удаление MiMo Code ==="
            export PATH="$HOME/.local/bin:$HOME/.mimo/bin:$PATH"
            if command -v mimo >/dev/null 2>&1; then
              mimo uninstall --force || true
            fi
            rm -rf "$HOME/.mimo" "$HOME/.config/mimo" 2>/dev/null || true

            echo "=== Удаление Ansible ==="
            if command -v apt-get >/dev/null 2>&1; then
              sudo apt-get remove -y ansible sshpass 2>/dev/null || true
              sudo apt-get autoremove -y 2>/dev/null || true
            fi
            python3 -m pip uninstall -y ansible 2>/dev/null || true
            rm -f "$HOME/.local/bin/ansible" "$HOME/.local/bin/ansible-playbook" 2>/dev/null || true

            echo "=== Проверка ==="
            command -v ansible >/dev/null 2>&1 && ansible --version || echo "ansible: removed"
            command -v mimo >/dev/null 2>&1 && mimo --version || echo "mimo: removed"
            """
        ).strip()
        # Выполнение скрипта удаления на целевом хосте
        result = _ssh_exec_on_target(cfg, uninstall_script, dry_run=dry_run)
        # Печать и возврат результата шага 8
        return _print_step("step_8: удаление Ansible и MiMo Code с целевого стенда", result)

    # Шаг 9: итоговая сводка по режиму работы скрипта
    def step_9_print_summary(
        # Конфигурация с данными целевого хоста
        cfg: dict[str, Any],
        # Запрет позиционной передачи mode и dry_run
        *,
        # Режим работы: install | uninstall | windows-only
        mode: str,
        # Был ли включён флаг --dry-run
        dry_run: bool,
    ) -> dict[str, str]:
        # Целевой хост для строки target в сводке
        target = cfg["target"]
        # Словарь итоговой информации и ссылок на документацию
        summary = {
            # Текущий режим выполнения скрипта
            "mode": mode,
            # Строковое представление флага dry-run
            "dry_run": str(dry_run),
            # Ожидаемый uname -a на прокси и стенде
            "expected_uname": TARGET_UNAME,
            # Целевой хост в формате user@host
            "target": f"{target['user']}@{target['host']}",
            # Ссылка на документацию MiMo Code
            "mimo_docs": "https://mimo.xiaomi.com/mimocode/install",
            # Команда установки MiMo на Windows
            "mimo_windows": "npm install -g @mimo-ai/cli",
            # Команда установки MiMo на Linux
            "mimo_linux": f"curl -fsSL {MIMO_INSTALL_URL} | bash",
            # Команда удаления MiMo на Linux
            "mimo_uninstall": "mimo uninstall --force",
            # Команда удаления Ansible через apt на Ubuntu
            "ansible_uninstall_ubuntu": "sudo apt-get remove -y ansible",
        }
        # Печать и возврат итоговой сводки
        return _print_step("step_9: итог", summary)

    # --- orchestration ---
    # Разбор аргументов командной строки (шаг 1)
    args = step_1_parse_args()
    # Загрузка конфигурации из YAML (шаг 2)
    cfg = step_2_load_config(args.config)
    # Вывод команд для Git Bash на Windows (шаг 3)
    step_3_print_windows_git_bash_commands(cfg)

    # Ранний выход: только инструкции для Windows без SSH
    if args.windows_only:
        # Итоговая сводка в режиме windows-only
        step_9_print_summary(cfg, mode="windows-only", dry_run=True)
        # Успешное завершение без развёртывания
        return 0

    # Предупреждение, если пароль прокси не задан и это не dry-run
    if not cfg["proxy"].get("password") and not args.dry_run:
        # Сообщение о необходимости PROXY_PASSWORD или proxy.password
        print(
            # Первая строка предупреждения о пароле прокси
            "ВНИМАНИЕ: задайте proxy.password в config.yaml "
            # Вторая строка — альтернатива через переменную окружения
            "или переменную окружения PROXY_PASSWORD\n"
        )

    # Проверка связности прокси → стенд (шаг 4)
    step_4_verify_connectivity(cfg, args.dry_run)

    # Ветка удаления ПО с целевого стенда
    if args.uninstall:
        # Удаление Ansible и MiMo (шаг 8)
        step_8_uninstall(cfg, args.dry_run)
        # Итог в режиме uninstall
        step_9_print_summary(cfg, mode="uninstall", dry_run=args.dry_run)
        # Успешное завершение после удаления
        return 0

    # Установка Ansible на стенде (шаг 5)
    step_5_install_ansible(cfg, args.dry_run)
    # Установка MiMo Code на стенде (шаг 6)
    step_6_install_mimo(cfg, args.dry_run)
    # Генерация Ansible inventory и cfg (шаг 7)
    step_7_write_ansible_inventory(cfg)
    # Итоговая сводка в режиме install
    step_9_print_summary(cfg, mode="install", dry_run=args.dry_run)
    # Успешное завершение полного цикла установки
    return 0


# Точка входа при запуске файла как python script.py
if __name__ == "__main__":
    # Обработка исключений с корректными кодами выхода
    try:
        # Запуск main() и передача её кода возврата в SystemExit
        raise SystemExit(main())
    # Отсутствующий config.yaml или другой FileNotFoundError
    except FileNotFoundError as exc:
        # Печать ошибки в stderr
        print(f"Ошибка: {exc}", file=sys.stderr)
        # Завершение с кодом 1
        raise SystemExit(1) from exc
    # Не установлен paramiko или другая зависимость
    except ImportError as exc:
        # Подсказка установить requirements.txt
        print(
            # Текст инструкции по установке зависимостей
            "Установите зависимости: python -m pip install -r requirements.txt",
            # Вывод сообщения в поток ошибок
            file=sys.stderr,
        )
        # Завершение с кодом 1
        raise SystemExit(1) from exc
    # Ошибки SSH, сети или RuntimeError из шагов проверки
    except (OSError, RuntimeError) as exc:
        # Печать ошибки выполнения в stderr
        print(f"Ошибка SSH/выполнения: {exc}", file=sys.stderr)
        # Завершение с кодом 1
        raise SystemExit(1) from exc
