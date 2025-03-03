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


async def download_file(url: str, dest_dir: Path, force: bool = False) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file: Path = dest_dir / os.path.basename(url)

    try:
        async with aiohttp.ClientSession(timeout=CLIENT_TIMEOUT) as session:
            async with session.get(url) as response:
                if response.status == HTTPStatus.OK:
                    total_size = int(response.headers.get("Content-Length", 0))  # Total file size
                    downloaded_size = 0

                    with dest_file.open("wb") as f:
                        async for chunk in response.content.iter_any():
                            f.write(chunk)
                            chunk_size = len(chunk)
                            downloaded_size += chunk_size

                            progress = (downloaded_size / total_size) * 100 if total_size else 0
                            sys.stdout.write(
                                f"\rDownload progress: {progress:.01f}%, "
                                f"{downloaded_size} of {total_size} bytes."
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
    args = parser.parse_args()

    if args.download_evaluations:
        asyncio.run(download_file(url=EVAL_URL, dest_dir=Path(DATA_EXTERNAL_DIR)))
    elif args.print_evaluations:
        eval_file_path: Path = Path(DATA_EXTERNAL_DIR) / os.path.basename(EVAL_URL)
        print_eval_file(path=eval_file_path)

if __name__ == "__main__":
    main()
