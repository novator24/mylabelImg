#!/usr/bin/env python3
"""
Развёртывание Ansible и MiMo Code на целевом Ubuntu-стенде через прокси-сервер.

Прокси: вход по логину/паролю.
Целевой хост: вход по SSH-ключам id_rsa/id_rsa.pub с прокси (~/.ssh/).

Документация MiMo Code: https://mimo.xiaomi.com/mimocode/install
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
EXAMPLE_CONFIG = SCRIPT_DIR / "config.example.yaml"

MIMO_INSTALL_URL = "https://mimo.xiaomi.com/install"
TARGET_UNAME = (
    "Linux team023 5.15.0-177-generic #187-Ubuntu SMP "
    "Sat Apr 11 22:54:33 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux"
)


def main() -> int:
    """Точка входа: установка или удаление Ansible и MiMo Code."""

    def _print_step(title: str, result: Any) -> Any:
        print(f"\n{'=' * 72}")
        print(f"  {title}")
        print(f"{'=' * 72}")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        elif isinstance(result, list):
            for line in result:
                print(f"  {line}")
        else:
            print(f"  {result}")
        print()
        return result

    def step_1_parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Развёртывание Ansible и MiMo Code через прокси-сервер",
        )
        parser.add_argument(
            "-c",
            "--config",
            default=str(DEFAULT_CONFIG),
            help=f"Путь к config.yaml (по умолчанию: {DEFAULT_CONFIG})",
        )
        parser.add_argument(
            "--uninstall",
            action="store_true",
            help="Удалить Ansible и MiMo Code с целевого стенда",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Только показать команды, без выполнения на серверах",
        )
        parser.add_argument(
            "--windows-only",
            action="store_true",
            help="Только вывести команды для Git Bash на Windows",
        )
        args = parser.parse_args()
        _print_step("step_1: аргументы CLI", vars(args))
        return args

    def step_2_load_config(config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            hint = (
                f"Файл {path} не найден. Скопируйте пример:\n"
                f"  cp {EXAMPLE_CONFIG} {DEFAULT_CONFIG}"
            )
            raise FileNotFoundError(hint)

        with path.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        proxy = cfg.setdefault("proxy", {})
        target = cfg.setdefault("target", {})
        cfg.setdefault("mimo", {})

        if not proxy.get("password"):
            proxy["password"] = os.environ.get("PROXY_PASSWORD", "")

        summary = {
            "config_file": str(path),
            "proxy": f"{proxy.get('user')}@{proxy.get('host')}:{proxy.get('port', 22)}",
            "target": f"{target.get('user')}@{target.get('host')}",
            "proxy_auth": "password" if proxy.get("password") else "PROXY_PASSWORD не задан",
            "target_key": target.get("private_key", "~/.ssh/id_rsa"),
            "mimo_url": cfg["mimo"].get("install_url", MIMO_INSTALL_URL),
        }
        cfg["_summary"] = summary
        _print_step("step_2: загрузка config.yaml", summary)
        return cfg

    def step_3_print_windows_git_bash_commands(cfg: dict[str, Any]) -> list[str]:
        """Команды для установки на локальной Windows-машине (Git Bash)."""
        target = cfg["target"]
        proxy = cfg["proxy"]

        commands = [
            "# --- Git Bash на Windows: подготовка ---",
            "# перейдите в каталог colab_c1 вашего репозитория",
            "cd colab_c1",
            "python -m pip install -r requirements.txt",
            "python --version",
            "node --version   # для MiMo Code нужен Node.js 18+",
            "",
            "# --- Ansible (через pip в Git Bash) ---",
            "python -m pip install --upgrade pip",
            "python -m pip install ansible",
            "ansible --version",
            "",
            "# --- MiMo Code (официальный способ для Windows) ---",
            "# https://mimo.xiaomi.com/mimocode/install",
            "npm install -g @mimo-ai/cli",
            "# при ошибке платформы:",
            "# npm install -g @mimo-ai/mimocode-windows-x64",
            "mimo --version",
            "",
            "# --- Конфиг: скопировать и заполнить ---",
            f"cp config.example.yaml config.yaml",
            "# задать PROXY_PASSWORD или proxy.password в config.yaml",
            "",
            "# --- Запуск скрипта развёртывания ---",
            "export PROXY_PASSWORD='your_proxy_password'",
            "python script.py --dry-run",
            "python script.py",
            "",
            "# --- Проверка SSH до прокси ---",
            f"ssh -p {proxy.get('port', 22)} {proxy['user']}@{proxy['host']}",
            "",
            "# --- С прокси: проверка ключей и доступа к стенду ---",
            "ls -la ~/.ssh/id_rsa ~/.ssh/id_rsa.pub",
            (
                f"ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=accept-new "
                f"{target['user']}@{target['host']} 'uname -a'"
            ),
            "",
            "# --- Ansible ping (после шага 7) ---",
            f"ansible -i {SCRIPT_DIR / 'inventory.ini'} stand -m ping",
            "",
            "# --- Удаление с Windows (локально) ---",
            "python -m pip uninstall -y ansible",
            "npm uninstall -g @mimo-ai/cli",
            "",
            "# --- Удаление с целевого Ubuntu-стенда ---",
            "python script.py --uninstall",
            "# MiMo на Linux: mimo uninstall --force",
        ]
        return _print_step("step_3: команды для Git Bash (Windows)", commands)

    def _ssh_exec_on_target(
        cfg: dict[str, Any],
        remote_command: str,
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        proxy = cfg["proxy"]
        target = cfg["target"]
        key = target.get("private_key", "~/.ssh/id_rsa")
        port = proxy.get("port", 22)

        escaped = remote_command.replace("'", "'\"'\"'")
        via_proxy = (
            f"ssh -i {key} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            f"{target['user']}@{target['host']} '{escaped}'"
        )

        if dry_run:
            return {
                "status": "dry-run",
                "via_proxy_command": via_proxy,
            }

        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=proxy["host"],
            port=port,
            username=proxy["user"],
            password=proxy.get("password") or None,
            look_for_keys=not proxy.get("password"),
            allow_agent=True,
            timeout=30,
        )
        try:
            _stdin, stdout, stderr = client.exec_command(via_proxy, get_pty=True)
            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="replace").strip()
            err = stderr.read().decode("utf-8", errors="replace").strip()
            return {
                "exit_code": exit_code,
                "stdout": out,
                "stderr": err,
                "command": via_proxy,
            }
        finally:
            client.close()

    def step_4_verify_connectivity(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        result = _ssh_exec_on_target(
            cfg,
            "uname -a && whoami && ls -la ~/.ssh/id_rsa ~/.ssh/id_rsa.pub",
            dry_run=dry_run,
        )
        if not dry_run and result.get("exit_code", 1) != 0:
            raise RuntimeError(
                f"Не удалось подключиться к целевому стенду через прокси:\n{result}"
            )
        return _print_step("step_4: проверка прокси → целевой Ubuntu", result)

    def step_5_install_ansible(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
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
        result = _ssh_exec_on_target(cfg, install_script, dry_run=dry_run)
        return _print_step("step_5: установка Ansible на целевом стенде", result)

    def step_6_install_mimo(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
        mimo_url = cfg.get("mimo", {}).get("install_url", MIMO_INSTALL_URL)
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
        result = _ssh_exec_on_target(cfg, install_script, dry_run=dry_run)
        return _print_step("step_6: установка MiMo Code на целевом стенде", result)

    def step_7_write_ansible_inventory(cfg: dict[str, Any]) -> dict[str, str]:
        target = cfg["target"]
        proxy = cfg["proxy"]
        key = target.get("private_key", "~/.ssh/id_rsa")
        proxy_jump = (
            f"-o ProxyJump={proxy['user']}@{proxy['host']}:{proxy.get('port', 22)}"
        )

        inventory_path = SCRIPT_DIR / "inventory.ini"
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
        inventory_path.write_text(inventory_content, encoding="utf-8")

        ansible_cfg_path = SCRIPT_DIR / "ansible.cfg"
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
        ansible_cfg_path.write_text(ansible_cfg_content, encoding="utf-8")

        result = {
            "inventory": str(inventory_path),
            "ansible_cfg": str(ansible_cfg_path),
            "ping_command": f"ansible -i {inventory_path} stand -m ping",
            "note": (
                "Запускайте ansible из каталога colab_c1 или укажите "
                f"ANSIBLE_CONFIG={ansible_cfg_path}"
            ),
        }
        return _print_step("step_7: конфиг Ansible (inventory + ansible.cfg)", result)

    def step_8_uninstall(cfg: dict[str, Any], dry_run: bool) -> dict[str, Any]:
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
        result = _ssh_exec_on_target(cfg, uninstall_script, dry_run=dry_run)
        return _print_step("step_8: удаление Ansible и MiMo Code с целевого стенда", result)

    def step_9_print_summary(
        cfg: dict[str, Any],
        *,
        mode: str,
        dry_run: bool,
    ) -> dict[str, str]:
        target = cfg["target"]
        summary = {
            "mode": mode,
            "dry_run": str(dry_run),
            "expected_uname": TARGET_UNAME,
            "target": f"{target['user']}@{target['host']}",
            "mimo_docs": "https://mimo.xiaomi.com/mimocode/install",
            "mimo_windows": "npm install -g @mimo-ai/cli",
            "mimo_linux": f"curl -fsSL {MIMO_INSTALL_URL} | bash",
            "mimo_uninstall": "mimo uninstall --force",
            "ansible_uninstall_ubuntu": "sudo apt-get remove -y ansible",
        }
        return _print_step("step_9: итог", summary)

    # --- orchestration ---
    args = step_1_parse_args()
    cfg = step_2_load_config(args.config)
    step_3_print_windows_git_bash_commands(cfg)

    if args.windows_only:
        step_9_print_summary(cfg, mode="windows-only", dry_run=True)
        return 0

    if not cfg["proxy"].get("password") and not args.dry_run:
        print(
            "ВНИМАНИЕ: задайте proxy.password в config.yaml "
            "или переменную окружения PROXY_PASSWORD\n"
        )

    step_4_verify_connectivity(cfg, args.dry_run)

    if args.uninstall:
        step_8_uninstall(cfg, args.dry_run)
        step_9_print_summary(cfg, mode="uninstall", dry_run=args.dry_run)
        return 0

    step_5_install_ansible(cfg, args.dry_run)
    step_6_install_mimo(cfg, args.dry_run)
    step_7_write_ansible_inventory(cfg)
    step_9_print_summary(cfg, mode="install", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ImportError as exc:
        print(
            "Установите зависимости: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except (OSError, RuntimeError) as exc:
        print(f"Ошибка SSH/выполнения: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
