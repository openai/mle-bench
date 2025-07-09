from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from appdirs import user_cache_dir

from mlebench.grade_helpers import Grader
from mlebench.utils import get_logger, get_module_dir, get_repo_dir, import_fn, load_yaml

logger = get_logger(__name__)


DEFAULT_DATA_DIR = (Path(user_cache_dir()) / "mle-bench" / "data").resolve()


@dataclass(frozen=True)
class Competition:
    id: str
    name: str
    description: str
    grader: Grader
    answers: Path
    gold_submission: Path
    sample_submission: Path
    competition_type: str
    prepare_fn: Callable[[Path, Path, Path], Path]
    raw_dir: Path
    private_dir: Path
    public_dir: Path
    checksums: Path
    leaderboard: Path

    def __post_init__(self):
        assert isinstance(self.id, str), "Competition id must be a string."
        assert isinstance(self.name, str), "Competition name must be a string."
        assert isinstance(self.description, str), "Competition description must be a string."
        assert isinstance(self.grader, Grader), "Competition grader must be of type Grader."
        assert isinstance(self.answers, Path), "Competition answers must be a Path."
        assert isinstance(self.gold_submission, Path), "Gold submission must be a Path."
        assert isinstance(self.sample_submission, Path), "Sample submission must be a Path."
        assert isinstance(self.competition_type, str), "Competition type must be a string."
        assert isinstance(self.checksums, Path), "Checksums must be a Path."
        assert isinstance(self.leaderboard, Path), "Leaderboard must be a Path."
        assert len(self.id) > 0, "Competition id cannot be empty."
        assert len(self.name) > 0, "Competition name cannot be empty."
        assert len(self.description) > 0, "Competition description cannot be empty."
        assert len(self.competition_type) > 0, "Competition type cannot be empty."

    @staticmethod
    def from_dict(data: dict) -> "Competition":
        grader = Grader.from_dict(data["grader"])

        try:
            return Competition(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                grader=grader,
                answers=data["answers"],
                sample_submission=data["sample_submission"],
                gold_submission=data["gold_submission"],
                competition_type=data["competition_type"],
                prepare_fn=data["prepare_fn"],
                raw_dir=data["raw_dir"],
                public_dir=data["public_dir"],
                private_dir=data["private_dir"],
                checksums=data["checksums"],
                leaderboard=data["leaderboard"],
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in competition config!")


class Registry:
    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self._data_dir = data_dir.resolve()

    def get_competition(self, competition_id: str) -> Competition:
        """Fetch the competition from the registry."""

        config_path = self.get_competitions_dir() / competition_id / "config.yaml"
        config = load_yaml(config_path)

        checksums_path = self.get_competitions_dir() / competition_id / "checksums.yaml"
        leaderboard_path = self.get_competitions_dir() / competition_id / "leaderboard.csv"

        description_path = get_repo_dir() / config["description"]
        description = description_path.read_text()

        preparer_fn = import_fn(config["preparer"])

        answers = self.get_data_dir() / config["dataset"]["answers"]
        gold_submission = answers
        if "gold_submission" in config["dataset"]:
            gold_submission = self.get_data_dir() / config["dataset"]["gold_submission"]
        sample_submission = self.get_data_dir() / config["dataset"]["sample_submission"]

        raw_dir = self.get_data_dir() / competition_id / "raw"
        private_dir = self.get_data_dir() / competition_id / "prepared" / "private"
        public_dir = self.get_data_dir() / competition_id / "prepared" / "public"

        return Competition.from_dict(
            {
                **config,
                "description": description,
                "answers": answers,
                "sample_submission": sample_submission,
                "gold_submission": gold_submission,
                "prepare_fn": preparer_fn,
                "raw_dir": raw_dir,
                "private_dir": private_dir,
                "public_dir": public_dir,
                "checksums": checksums_path,
                "leaderboard": leaderboard_path,
            }
        )

    def get_competitions_dir(self) -> Path:
        """Retrieves the competition directory within the registry."""

        return get_module_dir() / "competitions"

    def get_splits_dir(self) -> Path:
        """Retrieves the splits directory within the repository."""

        return get_repo_dir() / "experiments" / "splits"

    def get_lite_competition_ids(self) -> list[str]:
        """List all competition IDs for the lite version (low complexity competitions)."""

        lite_competitions_file = self.get_splits_dir() / "low.txt"
        with open(lite_competitions_file, "r") as f:
            competition_ids = f.read().splitlines()
        return competition_ids

    def get_data_dir(self) -> Path:
        """Retrieves the data directory within the registry."""

        return self._data_dir

    def set_data_dir(self, new_data_dir: Path) -> "Registry":
        """Sets the data directory within the registry."""

        return Registry(new_data_dir)

    def list_competition_ids(self) -> list[str]:
        """List all competition IDs available in the registry, sorted alphabetically."""

        competition_configs = self.get_competitions_dir().rglob("config.yaml")
        competition_ids = [f.parent.stem for f in sorted(competition_configs)]

        return competition_ids


registry = Registry()
