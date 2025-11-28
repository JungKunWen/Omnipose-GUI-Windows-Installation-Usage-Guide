# Omnipose GUI — Windows Installation & Usage Guide

A clean, reproducible, and stable setup guide for installing Omnipose (with GUI support) on Windows 10/11 using Anaconda.  
This guide is optimized for bacteria segmentation, high-throughput pipelines, and teaching environments.

Omnipose GUI is launched using:

```
omnipose
```

or:

```
python -m omnipose
```

---

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [1. Install Anaconda](#1-install-anaconda)
- [2. Create Omnipose Environment](#2-create-omnipose-environment)
- [3. Install Scientific Packages](#3-install-scientific-packages)
- [4. Install PyTorch (CPU or GPU)](#4-install-pytorch-cpu-or-gpu)
- [5. Install Omnipose and GUI Dependencies](#5-install-omnipose-and-gui-dependencies)
- [6. Launch Omnipose GUI](#6-launch-omnipose-gui)
- [7. Using Omnipose GUI for Bacteria Segmentation](#7-using-omnipose-gui-for-bacteria-segmentation)
- [8. Output Files](#8-output-files)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- Full GUI for interactive segmentation  
- Supports phase-contrast & fluorescence bacterial imaging  
- Optional GPU acceleration  
- Compatible with automated batch workflows  
- Stable Windows installation using verified dependency versions  

---

## System Requirements

**OS:**  
- Windows 10 / 11 (64-bit)

**Software:**  
- Anaconda 64-bit

**Optional hardware:**  
- NVIDIA GPU + updated driver (CUDA acceleration)

CPU mode is fully supported if GPU is not available.

---

## 1 Install Anaconda

Download from:  
https://www.anaconda.com/products/distribution

Use default settings.  
Do *not* manually add Anaconda to PATH.

---

## 2 Create Omnipose Environment

Open **Anaconda Prompt**:

```
conda create -n omni python=3.10.12 -y
conda activate omni
```

Prompt should show:

```
(omni) C:\Users\...
```
If "conda activate omni" fails, use the absolute activation command:
```
%USERPROFILE%\anaconda3\Scripts\activate %USERPROFILE%\anaconda3\envs\omni
```

---

## 3 Install Scientific Packages

Install verified compatible versions:

```
pip install "numpy==1.26.4" "scipy==1.10.1" "scikit-image==0.21.0" "tifffile==2023.7.10" --force-reinstall --ignore-installed llvmlite
pip install "opencv-python-headless<4.11" matplotlib tqdm pandas
```

---

## 4 Install PyTorch (CPU or GPU)

### CPU-only installation (recommended)

```
pip install torch==2.5.1+cpu torchvision --index-url https://download.pytorch.org/whl/cpu
```

### GPU installation (requires updated Nvidia driver)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Check CUDA availability

```
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

---

## 5 Install Omnipose and GUI Dependencies

### Install Omnipose:

```
pip install omnipose
```

### Install GUI dependencies (PyQt6 recommended)

```
pip install pyqt6 pyqt6-tools pyqt6-qt6 pyqt6-sip
pip install natsort
```

### Verify installation:

```
python -c "import omnipose; print('omnipose OK')"
```

---

## 6 Launch Omnipose GUI

Use:

```
omnipose
```

or:

```
python -m omnipose
```

A GUI window should appear.

---

## 7 Using Omnipose GUI for Bacteria Segmentation

### Step 1 — Load image  
File → Open images  
Supports TIFF / PNG / JPEG

### Step 2 — Select model  
Recommended for bacteria:

```
bact_phase_omni
```

### Step 3 — Set channels  
For single-channel phase images:

```
chan = 0
chan2 = 0
```

### Step 4 — Set diameter  
- `0` for auto-estimation  
- Or specify manually (e.g., 15–30 px)

### Step 5 — Run segmentation  
Check **Use GPU** if available.

### Step 6 — Save results  
Exports include masks, flows, outlines, npy arrays.

---

## 8 Output Files

Typical outputs:

```
image_cp_masks.tif
image_cp_flows.tif
image_cp_outlines.png
```

Useful for downstream measurement pipelines.

---

## Troubleshooting

### Missing natsort

```
pip install natsort
```

### GUI does not launch using python -m omnipose.gui  
Correct command:

```
omnipose
```

### GPU not detected

```
python -c "import torch; print(torch.cuda.is_available())"
```

If False → update Nvidia driver or use CPU PyTorch.

### Crash after clicking “Run”  
Usually caused by incompatible NumPy/SciPy.

Reinstall:

```
pip install "numpy==1.26.4" "scipy==1.10.1" "scikit-image==0.21.0" "tifffile==2023.7.10" --force-reinstall --ignore-installed llvmlite
```

---

## License

This guide may be included in laboratory GitHub repositories for internal educational and research purposes.
