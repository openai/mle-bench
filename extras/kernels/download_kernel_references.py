"""Script for downloading kernel references for competitions."""

import argparse

import kaggle

from mlebench.registry import registry
from mlebench.utils import get_logger

logger = get_logger(__name__)


def download_kernel_references(competition, max_kernels=50):
    """Downloads kernel references for a given competition."""
    kernels = kaggle.api.kernels_list(
        competition=competition.id, sort_by="voteCount", page_size=max_kernels
    )

    kernels_refs_path = registry.get_competitions_dir() / competition.id / "kernels.txt"
    kernels_refs_path.unlink(missing_ok=True)
    with kernels_refs_path.open(mode="a") as f:
        for kernel in kernels:
            kernel_ref = kernel.ref
            f.write(kernel_ref + "\n")


parser = argparse.ArgumentParser(
    description="Download kernel references for one or more competitions."
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-c",
    "--competition-id",
    help=(
        f"ID of the competition to download kernel references for. "
        f"Valid options: {registry.list_competition_ids()}"
    ),
    type=str,
)
group.add_argument(
    "-a",
    "--all",
    help="Download kernel references for all competitions.",
    action="store_true",
)
group.add_argument(
    "-l",
    "--list",
    help="Path to a text file listing competition IDs, one per line.",
    type=str,
)
parser.add_argument(
    "--max-kernels",
    help="Maximum number of kernels to download references for each competition.",
    type=int,
    default=50,
)

args = parser.parse_args()

if args.all:
    competition_ids = registry.list_competition_ids()
elif args.list:
    with open(args.list, "r") as f:
        competition_ids = f.read().splitlines()
else:
    competition_ids = [args.competition_id]

competitions = [registry.get_competition(cid) for cid in competition_ids]

for competition in competitions:
    download_kernel_references(competition, args.max_kernels)
