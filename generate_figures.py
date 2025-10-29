#!/usr/bin/env python3
"""
Run all figure-generating Python scripts in the current folder.

This script discovers `.py` files in the same directory (excluding itself and files
starting with an underscore) and executes them sequentially using the specified
Python interpreter. By default it will stop on the first failure; pass
`--continue-on-error` to run all scripts regardless of failures.

Usage:
    python generate_figures.py [--continue-on-error] [--python-executable /path/to/python]
"""
from pathlib import Path
import subprocess
import sys
import argparse
import logging


def find_scripts(folder: Path):
    scripts = [p for p in sorted(folder.iterdir())
               if p.suffix == '.py' and p.name != Path(__file__).name and not p.name.startswith('_')]
    return scripts


def run_script(python_executable: str, script_path: Path) -> int:
    logging.info(f"Running: {script_path.name}")
    try:
        # Run and stream output to the console
        proc = subprocess.run([python_executable, str(script_path)])
        return proc.returncode
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        raise
    except Exception as e:
        logging.exception(f"Failed to run {script_path}: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Run all .py figure scripts in this folder')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue running scripts after one fails')
    parser.add_argument('--python-executable', default=sys.executable, help='Python executable to use')
    parser.add_argument('--folder', default=None, help='Folder containing figure scripts (defaults to this script folder)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    folder = Path(args.folder).resolve() if args.folder else Path(__file__).resolve().parent
    folder = folder / "figure_scripts"
    
    logging.info(f"Discovering .py scripts in: {folder}")

    if not folder.exists():
        logging.error(f"Folder does not exist: {folder}")
        sys.exit(2)

    scripts = find_scripts(folder)
    if not scripts:
        logging.warning("No .py scripts found to run.")
        return

    failures = []
    for script in scripts:
        rc = run_script(args.python_executable, script)
        if rc != 0:
            logging.error(f"Script failed: {script.name} (exit {rc})")
            failures.append((script.name, rc))
            if not args.continue_on_error:
                logging.info("Stopping due to failure (use --continue-on-error to ignore errors)")
                break

    if failures:
        logging.info("Summary: Some scripts failed:")
        for name, rc in failures:
            logging.info(f"  {name}: exit {rc}")
        sys.exit(1)
    else:
        logging.info("All scripts completed successfully")


if __name__ == '__main__':
    main()
