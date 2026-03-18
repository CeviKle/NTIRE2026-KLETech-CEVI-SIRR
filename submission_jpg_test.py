import os
import re
import glob
import cv2
import numpy as np

src_root = "experiments/test_109/visualization/NTIRE2026-test"
dst_root = "submission_test109_q95"
jpg_q = 95

os.makedirs(dst_root, exist_ok=True)

# testing outputs use test_* naming
paths = sorted(glob.glob(src_root + "/test_*/*_clean_*.png"))
print("Found clean PNGs:", len(paths))

saved = 0
ids_seen = set()

for p in paths:
    base = os.path.basename(p)

    # match test_0001
    m = re.search(r"(test_\d{4})", base)
    if not m:
        print("WARN bad filename:", base)
        continue

    img_id = m.group(1)

    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("WARN cannot read:", p)
        continue

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    out_path = os.path.join(dst_root, f"{img_id}.jpg")
    ok = cv2.imwrite(
        out_path,
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_q)]
    )

    if ok:
        saved += 1
        ids_seen.add(img_id)
    else:
        print("WARN cannot write:", out_path)

print("Unique IDs:", len(ids_seen))
print("Saved JPGs:", saved)

if saved != 100:
    missing = [f"test_{i:04d}" for i in range(1, 101) if f"test_{i:04d}" not in ids_seen]
    print("MISSING COUNT:", len(missing))
    print("First missing:", missing[:20])
