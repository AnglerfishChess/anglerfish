#!/usr/bin/env python3
import argparse
import asyncio
import io
import json
import os
import sys
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Final

import aiohttp
import zstandard as zstd
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
                                f"\rDownload progress: {progress:.01f}%, {downloaded_size} of {total_size} byte(s)."
                            )
                            sys.stdout.flush()

                    print(f"\nDownloaded: {dest_file}")
                else:
                    print(f"\nFailed to download file. HTTP status code: {response.status}")

    except asyncio.TimeoutError:
        print("\nDownload timed out. Please try again later.")


def ui_print_eval_file(path: Path) -> None:
    """
    Lichess eval file is a Zstandard-compressed JSON file. Each line contains the JSON document for a single evaluation.

    The structure of each evaluation document::

        {
          "fen":          // the position FEN only contains pieces/active color/castling rights/en passant square.
          "evals": [      // a list of evaluations, ordered by number of PVs.
            "knodes":   // number of kilo-nodes searched by the engine
            "depth":    // depth reached by the engine
            "pvs": [    // list of principal variations
              "cp":     // centipawn evaluation. Omitted if mate is certain.
              "mate":   // mate evaluation. Omitted if mate is not certain.
              "line":   // principal variation, in UCI format.
            ]
          ]
        }

    Sample::

        {
          "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
          "evals": [
            {
              "pvs": [
                {
                  "cp": 311,
                  "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
                }
              ],
              "knodes": 206765,
              "depth": 36
            },
            {
              "pvs": [
                {
                  "cp": 292,
                  "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
                },
                {
                  "cp": 277,
                  "line": "f4g3 f7f5 e1e5 d8f6 a1e1 b7f7 g2c6 f8d8 d4d5 e6d5"
                }
              ],
              "knodes": 92958,
              "depth": 34
            },
            {
              "pvs": [
                {
                  "cp": 190,
                  "line": "h5h6 d8h4 h6g7 f8d8 f4g3 h4g4 c2e4 g4e4 g2e4 g8g7"
                },
                {
                  "cp": 186,
                  "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
                },
                {
                  "cp": 176,
                  "line": "f4g3 f7f5 e1e5 f5f4 g2e4 h7f6 e4b7 c8b7 g3f4 f6g4"
                }
              ],
              "knodes": 162122,
              "depth": 31
            }
          ]
        }

    :param path: path to the evaluation file.
    """
    if not path.exists():
        print(f"File not found: {path}")
        return

    lines = 0
    with path.open("rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                for line in text_reader:
                    lines += 1
                    if lines % 10000 == 0:
                        print(f"\rRead {lines} lines.", end="")

    print(f"\nTotal lines: {lines}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anglerfish CLI")
    parser.add_argument("--download-evaluations", action="store_true", help="Download the Lichess evaluations file.")
    parser.add_argument("--print-evaluations", action="store_true", help="Print the Lichess evaluations data.")
    parser.add_argument(
        "--train-network", action="store_true", help="Train the Pytorch network on the evaluations data."
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force the download without waiting or resuming.")
    args = parser.parse_args()

    if args.download_evaluations:
        asyncio.run(ui_download_file(url=EVAL_URL, dest_dir=Path(DATA_EXTERNAL_DIR), force=args.force))
    elif args.print_evaluations:
        eval_file_path: Path = Path(DATA_EXTERNAL_DIR) / os.path.basename(EVAL_URL)
        ui_print_eval_file(path=eval_file_path)
    elif args.train_network:
        eval_file_path: Path = Path(DATA_EXTERNAL_DIR) / os.path.basename(EVAL_URL)
        # This code creates and trains a Pytorch NN (Neural Network), using the following approach:
        # For each line (containing the evaluations as a JSON document), it takes the evaluation with the highest depth,
        # and uses its first PV as the "board evaluation".
        # The Pytorch NN uses the positions of the player pieces/pawns, and the opponent pieces/pawns,
        # as the input to the network; additionally, it uses some other chess-specific factors as the inputs
        # (for now, the only inputs should be "whether the player has a bishop pair of different colors: 1 bit;
        # whether the opponent has a bishop pair of different colors: 1 bit.").
        # The output of the network is the board evaluation (defined as we mentioned above).
        # The training is: get the board evaluation from the file; check the current evaluation according to the NN;
        # print how different it is (for user estimation); train the network on the evaluation from the file.
        # During the training process, it should dump the NN state (weights, whatever) to the file;
        # therefore whenever the training starts, it should check for existence of such a file and load it
        # (to be able to continue training).
        ui_train_network(path=eval_file_path)


if __name__ == "__main__":
    main()
