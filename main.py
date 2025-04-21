#!/usr/bin/env python3
"""
main.py ─────────────────────────────────────────────────────────────
Unified project entry point:
• Interactive selection between [Train / Infer]
• List available configs/*.yaml files for user selection
• Set environment variable CFG_FILE and dispatch the corresponding script
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


# ────────────────────────── Interactive functions ──────────────────────────
def _ask_mode() -> str:
    """Returns 'train' or 'infer'"""
    while True:
        ans = input("Select mode ([T]rain / [I]nfer) [T]: ").strip().lower()
        if ans in {"", "t", "train"}:
            return "train"
        if ans in {"i", "infer"}:
            return "infer"
        print("Invalid choice, please enter T or I.")


def _ask_cfg(cfg_dir: Path) -> str:
    """
    Prompt user to select a .yaml file in the configs directory.
    Returns the absolute path of the selected file.
    Selection of default.yaml is not allowed.
    """
    yaml_files = sorted(cfg_dir.glob("*.yaml"))

    # Exclude default.yaml
    yaml_files = [p for p in yaml_files if p.name != "default.yaml"]

    if not yaml_files:
        print("No valid configuration files found (excluding default.yaml).")
        sys.exit(1)

    print("\nAvailable configurations:")
    for idx, p in enumerate(yaml_files, 1):
        print(f"  {idx}) {p.name}")

    while True:
        ans = input(f"Select [1‑{len(yaml_files)}]: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(yaml_files):
            return str(yaml_files[int(ans) - 1])
        print("Invalid selection, try again.")


# ────────────────────────── Entry point ──────────────────────────
def main() -> None:
    proj_root = Path(__file__).resolve().parent
    cfg_dir = proj_root / "configs"

    mode = _ask_mode()
    cfg_path = _ask_cfg(cfg_dir)

    if cfg_path:
        os.environ["CFG_FILE"] = cfg_path
        print(f"\nOverride config set: {cfg_path}")

    # Allow importing from the "src" directory
    sys.path.insert(0, str(proj_root))

    module_name = "src.train" if mode == "train" else "src.infer"
    print(f"Launching {module_name}...\n")
    importlib.import_module(module_name).main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
