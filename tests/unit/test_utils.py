import tarfile

from mlebench.utils import extract, is_compressed


def test_extract_tar_gz_archive(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "data.txt"
    source_file.write_text("sample data")

    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_file, arcname="data.txt")

    destination = tmp_path / "destination"
    destination.mkdir()

    assert is_compressed(archive)

    extract(archive, destination)

    assert (destination / "data.txt").read_text() == "sample data"
