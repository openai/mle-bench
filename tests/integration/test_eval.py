from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from pytest import approx

from mlebench.data import download_and_prepare_dataset
from mlebench.grade import grade_csv
from mlebench.registry import Registry, registry
from mlebench.utils import in_ci
from tests.constants import fast_competitions, sample_submission_scores


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


@pytest.mark.slow
@pytest.mark.skipif(in_ci(), reason="Avoid slow-running tests in CI.")
@pytest.mark.parametrize(
    "competition_id, expected_score",
    sample_submission_scores.items(),
)
def test_sample_submission_for_competition_achieves_expected_score(
    competition_id: str,
    expected_score: float,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)
    report = grade_csv(competition.sample_submission, competition)
    actual_score = report.score

    if not np.isnan(expected_score):
        assert actual_score == approx(expected_score)
    else:
        assert np.isnan(actual_score)


@pytest.mark.parametrize(
    "competition_id, expected_score",
    [(c, s) for c, s in sample_submission_scores.items() if c in fast_competitions],
)
def test_sample_submission_for_fast_running_competition_achieves_expected_score(
    competition_id: str,
    expected_score: float,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)
    report = grade_csv(competition.sample_submission, competition)
    actual_score = report.score

    if not np.isnan(expected_score):
        assert actual_score == approx(expected_score)
    else:
        assert np.isnan(actual_score)


@pytest.mark.slow
@pytest.mark.skipif(in_ci(), reason="Avoid slow-running tests in CI.")
@pytest.mark.parametrize("competition_id", registry.list_competition_ids())
def test_submitting_answers_for_competition_achieves_a_gold_medal(
    competition_id: str,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)
    report = grade_csv(competition.gold_submission, competition)

    assert report.gold_medal


@pytest.mark.parametrize("competition_id", fast_competitions)
def test_submitting_answers_for_fast_running_competition_achieves_gold_medal(
    competition_id: str,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)
    report = grade_csv(competition.gold_submission, competition)

    assert report.gold_medal


@pytest.mark.slow
@pytest.mark.skipif(in_ci(), reason="Avoid slow-running tests in CI.")
@pytest.mark.parametrize("competition_id", registry.list_competition_ids())
def test_submitting_sample_submission_for_competition_doesnt_achieve_a_medal(
    competition_id: str,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)
    report = grade_csv(competition.sample_submission, competition)

    assert not report.any_medal


@pytest.mark.parametrize("competition_id", fast_competitions)
def test_submitting_sample_submission_for_fast_running_competition_doesnt_achieve_a_medal(
    competition_id: str,
    no_cache: bool,
    tmp_dir: Path,
):
    registry = Registry()

    if no_cache:
        registry = registry.set_data_dir(tmp_dir)

    competition = registry.get_competition(competition_id)
    download_and_prepare_dataset(competition)
    report = grade_csv(competition.sample_submission, competition)

    assert not report.any_medal
