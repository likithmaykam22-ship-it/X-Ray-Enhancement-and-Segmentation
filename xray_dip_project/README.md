# X-Ray DIP Lab — Enhancement & Segmentation

**X-Ray DIP Lab** is a Python-only desktop application based on the supplied *Medical X-Ray Enhancement and Segmentation* overview. It is intended for digital image processing coursework and research demonstrations, not clinical diagnosis.

## Project scope

The application implements the complete classical DIP pipeline described in the PDF:

| PDF stage | Application implementation |
|---|---|
| Raw X-ray image | Load PNG, JPG, BMP, TIFF, or other OpenCV-readable grayscale image |
| Preprocessing | None, 5×5 median filtering, or 5×5 Gaussian filtering |
| Enhancement | CLAHE, percentile contrast stretching, gamma correction, or unsharp masking |
| Segmentation | Otsu thresholding, adaptive thresholding, or Canny edge detection |
| Post-processing | Morphological opening, closing, or opening + closing |
| Output | Enhanced image, binary mask, green overlay, and optional saved files |
| Evaluation | Dice coefficient and Intersection over Union (IoU) against a loaded reference mask |

The implementation deliberately uses deterministic, explainable classical methods rather than a deep-learning model. This makes it suitable for a DIP subject project without requiring a GPU, model weights, or a backend service.

## Features

The interface provides a single workflow: load an X-ray, optionally load a binary ground-truth mask, choose preprocessing, enhancement, segmentation, and morphology settings, then run and inspect the result. The overlay view highlights segmented pixels in green on top of the enhanced image. Results can be exported as `original.png`, `enhanced.png`, `segmentation_mask.png`, and `overlay.png`.

The Dice and IoU values are displayed only when both a predicted mask and a reference mask are available. The reference image is resized to the input image dimensions using nearest-neighbor interpolation and binarized at intensity 127.

## Installation and run

Use Python 3.10 or newer on Windows, macOS, or Linux. From this folder, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python xray_dip_app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Build a Windows `.exe`

After installing the requirements on Windows, run:

```bat
build_exe.bat
```

The executable will be generated in `dist\XRayDIPLab\XRayDIPLab.exe`. The build uses PyInstaller in **onedir** mode because OpenCV and Pillow are more reliable in a folder distribution than in a single compressed executable. The entire `dist\XRayDIPLab` folder should be copied when distributing the application.

## Suggested coursework experiment

For a report, use one X-ray and compare at least three enhancement choices while keeping the segmentation method constant. Record the visual effect and, when a labelled mask is available, compare Dice and IoU. Then compare Otsu, adaptive thresholding, and Canny under the same enhancement setting. Discuss how noise, low contrast, and touching anatomical structures affect the output.

## Limitations and responsible use

This is an educational image-processing project. It does not identify disease, validate a clinical finding, or replace a radiologist. Classical thresholding and edge detection can fail when anatomy and background have similar intensity, when acquisition conditions vary, or when the target is not well represented by intensity boundaries. DICOM metadata and U-Net/Mask R-CNN inference are intentionally outside this lightweight Python-only version.

## Main files

| File | Purpose |
|---|---|
| `xray_dip_app.py` | Complete Tkinter GUI and OpenCV processing pipeline |
| `requirements.txt` | Python dependencies |
| `build_exe.bat` | Windows PyInstaller packaging script |
| `README.md` | Project documentation and coursework guidance |
