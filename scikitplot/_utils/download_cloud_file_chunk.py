"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""  # noqa: D404

import argparse as _argparse
import importlib as _importlib
import json as _json
import os as _os
import sys as _sys


def parse_args():
    """parse_args."""
    parser = _argparse.ArgumentParser()
    parser.add_argument("--range-start", required=True, type=int)
    parser.add_argument("--range-end", required=True, type=int)
    parser.add_argument("--headers", required=True, type=str)
    parser.add_argument("--download-path", required=True, type=str)
    parser.add_argument("--http-uri", required=True, type=str)
    return parser.parse_args()


def main():
    """Run main."""
    file_path = _os.path.join(_os.path.dirname(__file__), "request_utils.py")
    module_name = "mlflow.utils.request_utils"

    spec = _importlib.util.spec_from_file_location(module_name, file_path)
    module = _importlib.util.module_from_spec(spec)
    _sys.modules[module_name] = module
    spec.loader.exec_module(module)
    download_chunk = module.download_chunk

    args = parse_args()
    download_chunk(
        range_start=args.range_start,
        range_end=args.range_end,
        headers=_json.loads(args.headers),
        download_path=args.download_path,
        http_uri=args.http_uri,
    )


if __name__ == "__main__":
    main()
