"""
Download the OWID COVID-19 dataset into the repo root as owid-covid-data.csv.

The file is about 98 MB, so it isn't kept in git. OWID froze this dataset in
August 2024 (last date 2024-08-14), so the download is the same snapshot the
results in README.md were produced from.

Run from the repository root:  python scripts/get_data.py
"""
import hashlib
import os
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/owid/covid-19-data/master/"
       "public/data/owid-covid-data.csv")
DEST = "owid-covid-data.csv"
# SHA-256 of the file as downloaded on 17 Nov 2026 (98,391,483 bytes)
EXPECTED_SHA256 = "8473d0f0fdf962e1ffbd5b85b18726fc96a49bab109e271186c339725a12b10c"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if os.path.exists(DEST) and "--force" not in sys.argv:
        print(f"{DEST} already exists (use --force to re-download).")
        return
    print(f"Downloading {URL} ...")
    tmp = DEST + ".part"
    urllib.request.urlretrieve(URL, tmp)
    os.replace(tmp, DEST)
    digest = sha256(DEST)
    size_mb = os.path.getsize(DEST) / 1e6
    if digest == EXPECTED_SHA256:
        print(f"Saved {DEST} ({size_mb:.1f} MB), checksum matches.")
    else:
        print(f"Saved {DEST} ({size_mb:.1f} MB), but the checksum differs from the "
              f"snapshot used for the README ({digest[:12]}...). Results may not match exactly.")


if __name__ == "__main__":
    main()
