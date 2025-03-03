#!/usr/bin/env python3
import os
import sys
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Final

import aiohttp
import asyncio
import argparse

from aiohttp import ClientTimeout

EVAL_URL: Final[str] = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
DATA_EXTERNAL_DIR: Final[str] = "data-external"
CLIENT_TIMEOUT: Final[ClientTimeout] = aiohttp.ClientTimeout(total=timedelta(hours=1).total_seconds())


async def ui_download_file(url: str, dest_dir: Path, force: bool = False) -> None:
    """
    Downloads a file from the given URL and saves it to the specified destination directory.

    The function checks if the file already exists locally and compares its size to the remote file size.
    If the sizes match, it skips the download. If the sizes differ, the file is deleted and re-downloaded
    (but only after an interactive confirmation).
    Users can enforce a forced re-download (avoiding the confirmation) using the `force` parameter.

    :param url: The URL of the file to be downloaded.
    :param dest_dir: The destination directory where the file will be saved.
    :param force: A boolean flag to force re-downloading the file even if it exists locally.
    :return: None
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file: Path = dest_dir / os.path.basename(url)

    try:
        async with aiohttp.ClientSession(timeout=CLIENT_TIMEOUT) as session:
            # Analyzing the remote file size
            #
            async with session.head(url) as response:
                if response.status != HTTPStatus.OK:
                    print(f"\nFailed to fetch file details. HTTP status code: {response.status}")
                    return  # Failure
                else:
                    remote_size = int(response.headers.get("Content-Length", 0))
                    if dest_file.exists():
                        local_size = dest_file.stat().st_size
                        if local_size == remote_size:
                            # Case 1/3: File already downloaded completely
                            print("File already completely downloaded.")
                            return  # More-or-less good
                        else:
                            # Case 2/3: File exists but sizes do not match
                            print(
                                f"Existing file size: {local_size}, Remote file size: {remote_size}.\n"
                                f"File will be deleted and restarted downloading.\n"
                            )
                            if not force:
                                for i in range(10, 0, -1):
                                    sys.stdout.write(f"\rWill restart download in {i} second(s)...")
                                    sys.stdout.flush()
                                    await asyncio.sleep(1)
                                print()
                            # Delete the mismatched file before restarting the download
                            dest_file.unlink()
                            print("Deleted the mismatched file. Restarting download...")

                    else:
                        # Case 3/3: File does not exist
                        print("File does not exist locally; starting download...")

            # Downloading the file, if needed
            #
            async with session.get(url) as response:
                if response.status == HTTPStatus.OK:
                    total_size = int(response.headers.get("Content-Length", 0))  # Total file size
                    downloaded_size = 0

                    with dest_file.open("wb") as f:
                        async for chunk in response.content.iter_any():
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            progress = (downloaded_size / total_size) * 100 if total_size else 0
                            sys.stdout.write(
                                f"\rDownload progress: {progress:.01f}%, "
                                f"{downloaded_size} of {total_size} byte(s)."
                            )
                            sys.stdout.flush()

                    print(f"\nDownloaded: {dest_file}")
                else:
                    print(f"\nFailed to download file. HTTP status code: {response.status}")

    except asyncio.TimeoutError:
        print("\nDownload timed out. Please try again later.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anglerfish CLI")
    parser.add_argument("--download-evaluations", action="store_true", help="Download the Lichess evaluations file.")
    parser.add_argument("--print-evaluations", action="store_true", help="Print the Lichess evaluations data.")
    parser.add_argument(
        "--force", "-f", action="store_true", help="Force the download without waiting or resuming."
    )
    args = parser.parse_args()

    if args.download_evaluations:
        asyncio.run(ui_download_file(url=EVAL_URL, dest_dir=Path(DATA_EXTERNAL_DIR), force=args.force))
    elif args.print_evaluations:
        eval_file_path: Path = Path(DATA_EXTERNAL_DIR) / os.path.basename(EVAL_URL)
        print_eval_file(path=eval_file_path)


if __name__ == "__main__":
    main()
