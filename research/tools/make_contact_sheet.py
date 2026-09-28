"""生成带时间戳的联络表（contact sheet），用于人工重建区域真值。

用法: .venv/Scripts/python.exe research/tools/make_contact_sheet.py --frames-dir .tmp/seq_frames --out-dir .tmp/sheets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cell-w", type=int, default=480)
    args = ap.parse_args()

    files = sorted(args.frames_dir.glob("*.jpg"))
    times = json.loads((args.frames_dir / "times.json").read_text("utf-8"))
    times = times[: len(files)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import math
    per = args.cols * args.rows
    n_sheets = math.ceil(len(files) / per)
    for s in range(n_sheets):
        cells = []
        for k in range(per):
            i = s * per + k
            if i >= len(files):
                break
            im = cv2.imread(str(files[i]))
            h = int(round(args.cell_w * im.shape[0] / im.shape[1]))
            im = cv2.resize(im, (args.cell_w, h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(im, (0, 0), (args.cell_w, 34), (0, 0, 0), -1)
            cv2.putText(im, f"t={times[i]:.2f}s  #{i}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cells.append(im)
        rows = []
        for r in range(0, len(cells), args.cols):
            row = cells[r : r + args.cols]
            while len(row) < args.cols:
                row.append(np.full_like(cells[0], 30))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        p = args.out_dir / f"sheet_{s:02d}.jpg"
        cv2.imwrite(str(p), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print(f"[saved] {p}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
