# Project 1 — RGB Pixel Operations

This project explores the basics of **digital image processing** by explicitly scanning and manipulating RGB pixel values.  
The implementation avoids high-level image-processing shortcuts to reinforce a pixel-level understanding of how standard vision tasks work internally.

---

## Tech Stack
- **Language:** Python 3.11  
- **Environment:** Anaconda (virtual environment)  
- **Libraries:**  
  - [Pillow](https://pillow.readthedocs.io/) → image I/O and pixel manipulation  
  - [Matplotlib](https://matplotlib.org/) → visualization and plotting  
  - [PyTorch](https://pytorch.org/) (used for experimentation and validation of results)  

---

## Implementation Details
- All primary computations (channel extraction, grayscale averaging, histograms, thresholding, pyramids) are implemented via **explicit pixel-by-pixel scanning** with Python lists.  
- This satisfies the requirement of avoiding NumPy vectorization, OpenCV, or other high-level APIs.  
- PyTorch was used as a secondary tool to validate intermediate results against tensor-based implementations (e.g., histogram generation, thresholding).  

---

## Key Steps
1. **Load & Display Image** → Import a 512×512 JPEG as RGB (8-bit/channel).  
2. **Channel Isolation** → Extract Red, Green, and Blue channels:  
   - Grayscale single-channel representations.  
   - True RGB views where only one channel is preserved.  
3. **Grayscale Conversion (AG)** → Average of R, G, B at each pixel.  
4. **Histogram Computation** → Explicit 0–255 bin counting for RC, GC, BC, and AG.  
5. **Binarization (AB)** → Apply user-defined threshold TB (defaults to 128).  
6. **Edge Detection (AE)** → Forward-difference gradients (Gx, Gy), gradient magnitude, threshold TE.  
7. **Image Pyramid (AG2, AG4, AG8)** → Recursive downsampling with 2×2 block averaging.  

---

## Outputs
The program saves both raw images and figure plots for each stage:
- Channel isolations: `RC_gray.jpg`, `GC_gray.jpg`, `BC_gray.jpg`  
- True RGB channels: `RC_rgb.jpg`, `GC_rgb.jpg`, `BC_rgb.jpg`  
- Grayscale conversion: `AG.jpg`  
- Binary image: `AB.jpg`  
- Edge-detected image: `AE.jpg`  
- Image pyramid: `AG2.jpg`, `AG4.jpg`, `AG8.jpg`  
- Matplotlib plots (with titles/axes): saved as `*_plot.png`  

---

## Author
Naafiul Hossain

## Notes
- Last push on 8/31/25. Still a work in progress
