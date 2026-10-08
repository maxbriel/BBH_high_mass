#!/usr/bin/env python3
"""
Run all figure-generating Python scripts in figure_scripts/ and its sub-folders.

This script discovers `.py` files in figure_scripts/ and all sub-folders (excluding
files starting with an underscore and the scripts in EXCLUDE) and executes them
sequentially using the specified Python interpreter, each from its own folder so
their relative data paths resolve. By default it will stop on the first failure;
pass `--continue-on-error` to run all scripts regardless of failures.

Usage:
    python generate_figures.py [--continue-on-error] [--python-executable /path/to/python]
"""
from pathlib import Path
import os
import subprocess
import sys
import argparse
import logging

# scripts that cannot run from the data in data/
EXCLUDE = {
    "detectable_main_figure.py": "needs observable populations",
    "detectable_kick_figure.py": "needs observable populations",
}

REPO = Path(__file__).resolve().parent


def find_scripts(folder: Path):
    scripts = []
    for p in sorted(folder.rglob('*.py')):
        if p.name.startswith('_'):
            continue
        if p.name in EXCLUDE:
            logging.info(f"Skipping: {p.relative_to(folder)} ({EXCLUDE[p.name]})")
            continue
        scripts.append(p)
    return scripts


def run_script(python_executable: str, script_path: Path) -> int:
    logging.info(f"Running: {script_path.parent.name}/{script_path.name}")
    env = os.environ.copy()
    if not os.path.isdir(env.get("PATH_TO_POSYDON", "")):
        env["PATH_TO_POSYDON"] = str(REPO / "software" / "POSYDON")
    try:
        # Run from the script's folder and stream output to the console
        proc = subprocess.run([python_executable, script_path.name],
                              cwd=script_path.parent, env=env)
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
    # scripts run from their own folder, so a relative interpreter path must be absolute
    if os.sep in args.python_executable:
        args.python_executable = str(Path(args.python_executable).absolute())

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
