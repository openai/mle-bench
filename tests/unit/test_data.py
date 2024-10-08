from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mlebench.data import download_and_prepare_dataset, download_dataset
from mlebench.registry import Registry
from mlebench.utils import in_ci, is_empty
from tests.constants import fast_competitions


@pytest.fixture(scope="session")
def tmp_dir():
    """Creates a temporary directory."""

    tmp_dir = TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)

    yield tmp_path

    tmp_dir.cleanup()


@pytest.fixture
def no_cache(request):
    return request.config.getoption("--no-cache")


@pytest.mark.skipif(in_ci(), reason="Avoid rate limits in CI.")
@pytest.mark.parametrize("competition_id", fast_competitions)
def test_downloading_competition_dataset_downloads_a_zip_file(
    competition_id: str,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    download_dir = registry.get_data_dir() / competition_id
    zipfile = download_dataset(competition_id=competition_id, download_dir=download_dir)

    assert zipfile.is_file()
    assert zipfile.suffix == ".zip"


@pytest.mark.skipif(in_ci(), reason="Avoid rate limits in CI.")
@pytest.mark.parametrize("competition_id", fast_competitions)
def test_preparing_competition_dataset_creates_non_empty_public_and_private_directories(
    competition_id: str,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)

    assert competition.public_dir.is_dir(), f"Public directory does not exist!"
    assert not is_empty(competition.public_dir), f"Public directory is empty!"
    assert competition.private_dir.is_dir(), f"Private directory does not exist!"
    assert not is_empty(competition.private_dir), f"Private directory is empty!"
