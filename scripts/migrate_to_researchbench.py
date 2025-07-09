#!/usr/bin/env python
"""
Script to help users migrate from mlebench to researchbench.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_mlebench_installation():
    """Check if mlebench is installed."""
    try:
        import mlebench
        return True
    except ImportError:
        return False


def check_researchbench_installation():
    """Check if researchbench is installed."""
    try:
        import researchbench
        return True
    except ImportError:
        return False


def migrate_data(mlebench_data_dir, researchbench_data_dir):
    """Migrate data from mlebench to researchbench."""
    if not mlebench_data_dir.exists():
        print(f"mlebench data directory {mlebench_data_dir} does not exist. Skipping data migration.")
        return
    
    if not researchbench_data_dir.exists():
        researchbench_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy competition data
    for item in mlebench_data_dir.iterdir():
        if item.is_dir():
            target = researchbench_data_dir / item.name
            if not target.exists():
                print(f"Copying {item} to {target}...")
                shutil.copytree(item, target)
            else:
                print(f"{target} already exists. Skipping.")
        elif item.is_file():
            target = researchbench_data_dir / item.name
            if not target.exists():
                print(f"Copying {item} to {target}...")
                shutil.copy2(item, target)
            else:
                print(f"{target} already exists. Skipping.")


def install_researchbench(repo_dir):
    """Install researchbench."""
    print("Installing researchbench...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)], check=True)


def main():
    parser = argparse.ArgumentParser(description="Migrate from mlebench to researchbench.")
    parser.add_argument(
        "--mlebench-data-dir",
        type=str,
        default=str(Path.home() / ".cache" / "mle-bench" / "data"),
        help="Path to the mlebench data directory.",
    )
    parser.add_argument(
        "--researchbench-data-dir",
        type=str,
        default=str(Path.home() / ".researchbench" / "data"),
        help="Path to the researchbench data directory.",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Path to the repository directory.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installation of researchbench.",
    )
    parser.add_argument(
        "--skip-data-migration",
        action="store_true",
        help="Skip migration of data from mlebench to researchbench.",
    )
    args = parser.parse_args()

    # Convert paths to Path objects
    mlebench_data_dir = Path(args.mlebench_data_dir)
    researchbench_data_dir = Path(args.researchbench_data_dir)
    repo_dir = Path(args.repo_dir)
    
    # Check if mlebench is installed
    mlebench_installed = check_mlebench_installation()
    if mlebench_installed:
        print("mlebench is installed.")
    else:
        print("mlebench is not installed.")
    
    # Check if researchbench is installed
    researchbench_installed = check_researchbench_installation()
    if researchbench_installed:
        print("researchbench is already installed.")
    else:
        print("researchbench is not installed.")
        if not args.skip_install:
            install_researchbench(repo_dir)
    
    # Migrate data
    if not args.skip_data_migration:
        migrate_data(mlebench_data_dir, researchbench_data_dir)
    
    print("\nMigration completed successfully!")
    print("\nYou can now use researchbench instead of mlebench. For example:")
    print("  - Instead of 'mlebench prepare --all', use 'researchbench prepare --all'")
    print("  - Instead of 'mlebench grade-sample <PATH> <ID>', use 'researchbench grade-sample <PATH> <ID>'")
    print("\nResearchBench also includes new functionality for research tasks:")
    print("  - List research tasks: 'researchbench research list'")
    print("  - Prepare a research task: 'researchbench research prepare <task-id>'")
    print("  - Run a research task: 'researchbench research run <task-id>'")
    print("\nSee the README.md file for more information.")


if __name__ == "__main__":
    main()