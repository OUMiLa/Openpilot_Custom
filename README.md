# Openpilot Dataset Preparation Project

This repository contains the **modified openpilot files** and **guide** you need to complete a dataset preparation project. Your goal is to turn an openpilot driving replay into a structured dataset with images, model features, depth maps, and object detections.

---

## 📋 What This Repository Contains

This repo contains **ONLY**:

- **`docs/DATA_PREPARATION_GUIDE.md`** — Step-by-step guide explaining what you need to do
- **`openpilot_files/selfdrive/modeld_detection_first.py`** — Modified modeld file (copy this into your openpilot)
- **`openpilot_files/selfdrive/modeld_detection_second.py`** — Alternative modified modeld file (copy this into your openpilot)

**You will need to provide:**
- Your own openpilot repository checkout
- YOLO (for object detection)
- Depth Anything V2 (for depth estimation)

---

## 🚀 Quick Start

### Step 1: Clone openpilot (separately)

```bash
cd ~
git clone https://github.com/commaai/openpilot.git
cd openpilot
```

Follow openpilot's setup instructions to get your environment ready.

### Step 2: Copy the modified files into your openpilot

```bash
# From this repository, copy the modeld files into your openpilot checkout
cp openpilot_files/selfdrive/modeld_detection_second.py ~/openpilot/selfdrive/modeld/
```

**Note:** You can use either `modeld_detection_first.py` or `modeld_detection_second.py`. The second version includes automatic segment management (saves to `segment_00`, `segment_01`, etc.).

### Step 3: Follow the guide

Read **`docs/DATA_PREPARATION_GUIDE.md`** for the complete workflow:

1. **Replay a route** using openpilot's replay tool
2. **Capture images + features** using the modified modeld file
3. **Run offline labeling:**
   - Use **YOLO** to detect objects (cars, pedestrians, etc.)
   - Use **Depth Anything V2** to estimate depth maps
   - Merge detections + depth into training-ready labels

---

## 📚 What You'll Learn

- How openpilot's vision model works (feature extraction)
- How to capture training data from replays
- How to use modern CV tools (YOLO, depth estimation) for labeling
- How to organize datasets for machine learning

---

## ✅ Deliverables

At the end, you should have:

- A dataset folder with multiple segments (`segment_00`, `segment_01`, ...)
- Each segment containing:
  - `raw/` — camera images
  - `features/` — model feature embeddings
  - `depth_npy/` — depth maps (if you ran depth estimation)
  - `labels/` — object detection labels (if you ran YOLO)
- A merged label file combining detections + depth statistics
- A short report summarizing your dataset

---

## 🛠️ Prerequisites

- Linux environment (Ubuntu recommended)
- Python 3 with openpilot dependencies
- GPU (recommended, for faster depth + detection)
- Enough disk space (datasets can be large)
- **YOLO** installed (e.g., `pip install ultralytics`)
- **Depth Anything V2** repository cloned and set up

---

## 📖 Full Instructions

See **`docs/DATA_PREPARATION_GUIDE.md`** for the complete step-by-step guide.

---

## ❓ Troubleshooting

If you encounter issues:

1. Make sure your openpilot environment is set up correctly
2. Verify the modified modeld file is in the right location
3. Check that replay is running and publishing camera frames
4. See the troubleshooting section in `docs/DATA_PREPARATION_GUIDE.md`

---

## 📝 Notes

- The modified `modeld` files save dataset frames/features but otherwise behave like normal openpilot modeld
- Depth estimation produces **relative depth** (not true meters) unless you add calibration
- Object detection confidence scores are **not distance** — combine with depth for distance estimates
