# NTIRE 2026: Reflection Removal in the Wild (SIRR)

## 1. Introduction

This repository contains our implementation for the **NTIRE 2026 Reflection Removal in the Wild Challenge**, conducted as part of the CVPR 2026 Workshop. The task focuses on recovering clean transmission images from real-world scenes affected by reflections.

Our implementation is built upon the XReflection framework and adapted for challenge-specific training, validation, and inference.

---

## 2. Method Overview

We adopt a CNN-based dual network architecture for reflection removal. The model is trained to separate reflection and transmission components from a single input image.

* Backbone: XReflection
* Training: Supervised learning
* Total training duration: **112 epochs**
* Evaluation: PSNR-based checkpoint comparison

---

## 3. Repository Structure

```
options/                Configuration files (train / val / test)
xreflection/            Model + training pipeline
experiments/            Outputs, logs, checkpoints
make_submission_jpg.py  Submission script
rotate_ensemble.py      Ensemble utility
```

---

## 4. Installation

```
git clone https://github.com/CeviKle/NTIRE2026-KLETech-CEVI-SIRR.git
cd NTIRE2026-KLETech-CEVI-SIRR
pip install -r requirements.txt
```

---

## 5. GPU Usage

* **GPU 0 → Training**
* **GPU 1 → Validation / Testing**

This ensures training and evaluation can run independently without interference.

---

## 6. Training

### 6.1 Start Training

```
CUDA_VISIBLE_DEVICES=0 python xreflection/tools/train.py \
  --config options/train_rdnet.yml
```

### 6.2 Resume Training

```
CUDA_VISIBLE_DEVICES=0 python xreflection/tools/train.py \
  --config options/train_rdnet.yml \
  --resume experiments/train_sirs_rdnet/checkpoints/last.ckpt
```

---

## 7. Validation Setup

### 7.1 Important Note

We **do not modify the original training YAML**.
Instead, we create separate validation configurations to avoid affecting training.

---

## 7.2 Creating Validation Config (val_300)

```
python - <<'PY'
import yaml

src = "options/train_rdnet.yml"
dst = "options/train_rdnet_val300.yml"

with open(src, "r") as f:
    cfg = yaml.safe_load(f)

cfg["datasets"]["val_datasets"][0]["datadir"] = "/NTIRE2026/C1_ReflRem/val_300"
cfg["datasets"]["val_datasets"][0]["mode"] = "eval"
cfg["test_only"] = True

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote:", dst)
PY
```

---

## 7.3 Creating Checkpoint-Specific Config (Example: epoch 94)

Instead of writing a new config from scratch, we copy an existing one:

```
cp options/val_ep17_only.yml options/val_ep94_only.yml
```

Then update its name:

```
python - <<'PY'
import yaml
p = "options/val_ep94_only.yml"
cfg = yaml.safe_load(open(p))
cfg["name"] = "val_ep94_only"
yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
print("OK name =", cfg["name"])
PY
```

---

## 7.4 Run Validation (val_300)

```
CUDA_VISIBLE_DEVICES=1 python xreflection/tools/train.py \
  --config options/train_rdnet_val300.yml \
  --test_only experiments/train_sirs_rdnet/checkpoints/last.ckpt
```

---

## 7.5 Evaluate Specific Checkpoint

```
CUDA_VISIBLE_DEVICES=1 python xreflection/tools/train.py \
  --config options/train_rdnet_val300.yml \
  --test_only experiments/train_sirs_rdnet/checkpoints/epoch=94-step=XXXXX.ckpt
```

---

## 8. Validation Note

* Validation dataset contains only **blended images**
* We used **pseudo / approximate ground truth**
* Therefore:

  * PSNR values are **not absolute**
  * Used only for **relative comparison**

---

## 9. Output Location

Outputs are saved in:

```
experiments/train_sirs_rdnet/
```

Find outputs:

```
find experiments/train_sirs_rdnet -type f | grep -iE "\.png$|\.jpg$"
```

Count outputs:

```
find experiments/train_sirs_rdnet -type f | grep -iE "\.png$|\.jpg$" | wc -l
```

Expected:

* Validation → 300 images
* Test → 100 images

---

## 10. Test Inference

```
CUDA_VISIBLE_DEVICES=1 python xreflection/tools/train.py \
  --config options/test_100.yml \
  --test_only experiments/train_sirs_rdnet/checkpoints/last.ckpt
```

---

## 11. Submission

Prepare:

* val_300_output.zip → 300 images
* test_100_output.zip → 100 images

Ensure:

* Correct naming
* No extra folders
* No missing images

---

## 12. Acknowledgement

This work is based on:
https://github.com/hainuo-wang/XReflection

---
