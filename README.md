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
