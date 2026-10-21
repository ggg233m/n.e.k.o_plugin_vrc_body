# -*- coding: utf-8 -*-
"""Resumable downloader for Middlebury 2014 (via local HTTP proxy). ASCII only."""
import os
import sys
import time
import urllib.request as ur

BASE = "https://vision.middlebury.edu/stereo/data/scenes2014/zip/"
OUT = os.path.join(".tmp", "mb2014")
PROXY = "http://127.0.0.1:7897"

SCENES = ["Adirondack", "Bicycle1", "Classroom1", "Couch", "Flowers",
          "Motorcycle", "Piano", "Playtable", "Recycle", "Shelves"]

opener = ur.build_opener(ur.ProxyHandler({"http": PROXY, "https": PROXY}))


def size_of(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


def total_size(url):
    req = ur.Request(url, method="HEAD")
    with opener.open(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(name, want=None):
    path = os.path.join(OUT, name + "-perfect.zip")
    url = BASE + name + "-perfect.zip"
    if want is None:
        want = total_size(url)
    if size_of(path) >= want:
        return want, want, "cached"
    have = size_of(path)
    t0 = time.time()
    last = have
    stall = 0
    with open(path, "ab") as f:
        while have < want:
            try:
                req = ur.Request(url, headers={"Range": "bytes=%d-" % have})
                with opener.open(req, timeout=90) as r:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                f.flush()
            except Exception as e:
                f.flush()
                have = size_of(path)
                if have == last:
                    stall += 1
                else:
                    stall = 0
                last = have
                print("  retry %s at %d/%d (%s)" % (name, have, want, type(e).__name__))
                if stall >= 40:
                    print("  GIVING UP %s at %d" % (name, have))
                    break
                time.sleep(min(2 + stall, 10))
    return size_of(path), want, "%.1fs" % (time.time() - t0)


def main():
    os.makedirs(OUT, exist_ok=True)
    for n in SCENES:
        got, want, note = fetch(n)
        print("%-12s %d/%d  %s" % (n, got, want, note))
    print("ALLDONE")


if __name__ == "__main__":
    main()
