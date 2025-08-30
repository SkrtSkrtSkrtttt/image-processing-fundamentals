This project explores the basics of image processing by scanning and manipulating RGB pixel values directly.  
It demonstrates how to isolate color channels, compute grayscale, generate histograms, apply thresholding, detect edges, and build an image pyramid.  
The goal is to understand how pixel-level operations work under the hood without relying on high-level image processing libraries.

---



Key steps implemented:
1. Load and display a 512×512 RGB image.
2. Isolate **Red, Green, Blue** channels (both grayscale and true-color views).
3. Convert to grayscale by averaging R, G, B.
4. Compute histograms for RC, GC, BC, and grayscale.
5. Perform **binarization** with a user-defined threshold.
6. Perform **edge detection** using forward differences and gradient magnitude.
7. Construct an **image pyramid** (AG2, AG4, AG8) by downsampling with 2×2 averaging
