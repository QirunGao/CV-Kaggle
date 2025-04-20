#!/usr/bin/env python3
"""
main.py ─────────────────────────────────────────────────────────────
项目统一入口：
• 交互式选择【Train / Infer】
• 列出 configs/*.yaml 让用户选择覆盖配置
• 设置环境变量 CFG_FILE 并调度对应脚本
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


# ────────────────────────── 交互函数 ──────────────────────────
def _ask_mode() -> str:
    """返回 'train' 或 'infer'"""
    while True:
        ans = input("Select mode ([T]rain / [I]nfer) [T]: ").strip().lower()
        if ans in {"", "t", "train"}:
            return "train"
        if ans in {"i", "infer"}:
            return "infer"
        print("❌  Invalid choice, please enter T or I.")


def _ask_cfg(cfg_dir: Path) -> str:
    """
    让用户选择 configs 目录下的 yaml 文件；
    返回选中文件的绝对路径，强制要求选择非 default.yaml 的配置文件。
    """
    yaml_files = sorted(cfg_dir.glob("*.yaml"))

    # 过滤掉 default.yaml
    yaml_files = [p for p in yaml_files if p.name != "default.yaml"]

    if not yaml_files:
        print("❌  No valid configuration files found (excluding default.yaml).")
        sys.exit(1)

    print("\nAvailable config:")
    for idx, p in enumerate(yaml_files, 1):
        print(f"  {idx}) {p.name}")

    while True:
        ans = input(f"Select [1‑{len(yaml_files)}]: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(yaml_files):
            return str(yaml_files[int(ans) - 1])
        print("❌  Invalid selection, try again.")


# ────────────────────────── 入口 ──────────────────────────
def main() -> None:
    proj_root = Path(__file__).resolve().parent
    cfg_dir = proj_root / "configs"

    mode = _ask_mode()
    cfg_path = _ask_cfg(cfg_dir)

    if cfg_path:
        os.environ["CFG_FILE"] = cfg_path
        print(f"\n✅  Override config set: {cfg_path}")

    # 让 “src” 能被 import
    sys.path.insert(0, str(proj_root))

    module_name = "src.train" if mode == "train" else "src.infer"
    print(f"🚀  Launching {module_name} …\n")
    importlib.import_module(module_name).main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
