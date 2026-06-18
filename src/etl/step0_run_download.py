"""Stage 0 of the ETL pipeline: download raw competition data from Kaggle.

Reads the competition slug and target directory from ``configs/etl.yaml``
and downloads the raw files into the configured ``dirs.raw`` directory.
If the target already contains data files the download is skipped unless
``--force`` is given. Any downloaded zip archives are extracted and removed
when ``unzip`` is enabled in the config.

Credentials are read from the ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``
environment variables, which may be supplied via a ``.env`` file at the
project root (loaded through python-dotenv).

Typical usage example:

    $ uv run python -m src.etl.step0_run_download
    $ uv run python -m src.etl.step0_run_download --force
"""
import argparse
import zipfile

from dotenv import load_dotenv
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

from src.etl._config import load_config  # noqa: E402


def main(force: bool = False):
    """Download the raw Kaggle dataset into the configured target directory.

    Creates the target directory if it does not exist, then downloads the
    competition files unless data is already present.

    Args:
        force: If True, re-download even when the target directory already
            contains data files. Defaults to False, in which case a
            non-empty target (ignoring ``.gitkeep``) skips the download.

    Raises:
        OSError: If Kaggle credentials are missing or invalid, i.e. neither
            the ``KAGGLE_USERNAME``/``KAGGLE_KEY`` environment variables nor
            ``~/.kaggle/kaggle.json`` are available when authenticating.
    """
    config = load_config()
    download_cfg = config["download"]
    target = config["dirs"]["raw"]
    if not target.exists():
        print(f"Creating directory: {target}")
        target.mkdir(parents=True)

    existing = [p for p in target.iterdir() if p.is_file() and p.name != ".gitkeep"]
    if existing and not force:
        print(f"Found {len(existing)} file(s) in {target}, skip download. (use --force to re-download)")
        return

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading '{download_cfg['competition']}' → {target}")
    api.competition_download_files(download_cfg["competition"], path=str(target))

    if download_cfg.get("unzip"):
        for zip_path in target.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(target)
            zip_path.unlink() 
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 Kaggle 下載原始資料到 data/raw/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 data/raw 已有檔案也強制重新下載（預設 False）",
    )
    args = parser.parse_args()
    main(force=args.force)
