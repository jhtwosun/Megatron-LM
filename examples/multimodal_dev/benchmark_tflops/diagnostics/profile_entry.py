"""Explicit opt-in wrapper; caller supplies the qualified training entry and arguments."""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

from causal_ranges import install, receipt
from clock_brackets import calibrate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    entry = args.entry.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)
    if not output_dir.is_dir():
        parser.error("--output-dir must already exist")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16:
        parser.error("this diagnostic is qualified only for world size 16")
    output = output_dir / f"causal-clock-{rank:05d}.json"
    # Fail before training on an existing receipt; final creation stays exclusive.
    if output.exists():
        raise FileExistsError(output)
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    install()
    pre = calibrate("pre")
    sys.argv = [str(entry), *training_args]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
    post = calibrate("post")
    result = dict(
        clock=dict(pre=pre, post=post),
        identity=receipt(),
        training_entry=str(entry),
        training_args=training_args,
    )
    with output.open("x") as stream:
        json.dump(result, stream)
    print(
        "CAUSAL_CLOCK_FILE "
        + json.dumps(
            dict(
                rank=rank, path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest()
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
