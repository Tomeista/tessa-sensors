"""Download NASA C-MAPSS (turbofan engine degradation) into data/raw/CMAPSSData/.

Usage:  python data/download.py

The raw data is not committed to git. The archive comes from the NASA Prognostics
Center of Excellence data repository; the SHA-256 hashes below make sure everyone
in the group trains on byte-identical files.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
RAW = Path(__file__).resolve().parent / "raw" / "CMAPSSData"
FILES = [f"{kind}_{fd}.txt" for fd in ("FD001", "FD002", "FD003", "FD004") for kind in ("train", "test", "RUL")]
SHA256 = {
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> bool:
    ok = True
    for name in FILES:
        path = RAW / name
        if not path.exists():
            print(f"missing: {name}")
            ok = False
        elif SHA256.get(name) and sha256(path) != SHA256[name]:
            print(f"hash mismatch: {name}")
            ok = False
    return ok


def main() -> int:
    if all((RAW / f).exists() for f in FILES) and verify():
        print(f"C-MAPSS already present in {RAW}")
        return 0
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    # The NASA archive wraps the actual data in a second zip file.
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    RAW.mkdir(parents=True, exist_ok=True)
    for name in inner.namelist():
        base = Path(name).name  # flatten paths, never write outside RAW
        if base in FILES or base == "readme.txt":
            (RAW / base).write_bytes(inner.read(name))
    if not verify():
        return 1
    print(f"C-MAPSS written to {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
