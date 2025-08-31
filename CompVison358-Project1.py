# Naafiul Hossain — Project 1: RGB Pixel Operations (pixel-scanned)
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# INPUT CONFIG
# Set the path to the 512×512 JPEG test image (e.g., shed1-small.jpg).
# You may replace this with a relative path like "shed1-small.jpg" if the file
# lives next to this script.
# -----------------------------------------------------------------------------
IMAGE_PATH = r"C:\Users\Naafiul Hossain\Downloads\shed1-small (1).jpg"

# -----------------------------------------------------------------------------
# UTILS
# - All helpers keep data in simple Python lists (H×W) to satisfy the
#   requirement of scanning/processing pixels explicitly (no high-level ops).
# -----------------------------------------------------------------------------
def load_rgb_u8(path):
    """
    Open an image file, force RGB (8 bits/channel), and validate size.
    The assignment requires exactly 512×512 input with 8-bit samples.
    """
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        raise ValueError(f"Image must be 512x512, got {img.size}. Use shed1-small.jpg/shed2-small.jpg.")
    return img

def show_img(img, title):
    """
    Display an image with Matplotlib. If it's a single-band (grayscale) PIL
    image ('L' mode), render it with a true grayscale colormap to avoid
    false-color (viridis) visuals.
    """
    if isinstance(img, Image.Image) and img.mode == "L":
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def pil_gray_from_2d(arr2d):
    """
    Convert a 2D list of integers in [0,255] (H×W) into a PIL grayscale image.
    This is used after we compute results via explicit pixel scans.
    """
    h, w = len(arr2d), len(arr2d[0])
    out = Image.new("L", (w, h))
    out.putdata([v for row in arr2d for v in row])
    return out

def pil_from_2d(arr2d, mode="L"):
    """
    Generic converter from 2D list to a PIL image with the given mode.
    For this project we only use "L" (grayscale) for channel slices.
    """
    h, w = len(arr2d), len(arr2d[0])
    out = Image.new(mode, (w, h))
    out.putdata([v for row in arr2d for v in row])
    return out

def save_gray(path, arr2d):
    """Save a 2D list as a grayscale JPEG on disk."""
    pil_gray_from_2d(arr2d).save(path, format="JPEG")

# -----------------------------------------------------------------------------
# 1) LOAD & SHOW ORIGINAL
# -----------------------------------------------------------------------------
A = load_rgb_u8(IMAGE_PATH)
show_img(A, "Original Image A (512×512)")

# Create a fast random-access handle: pixels[x, y] -> (R, G, B)
W, H = A.size
pixels = A.load()

# -----------------------------------------------------------------------------
# 2) CHANNEL ISOLATION
# Build per-channel 2D arrays (H×W). We read each pixel once and split R/G/B.
# Then we render two versions:
#   - Grayscale single-band images (for histograms and clarity)
#   - True RGB images where two channels are zeroed (so red looks truly red)
# -----------------------------------------------------------------------------
RC, GC, BC = [[0]*W for _ in range(H)], [[0]*W for _ in range(H)], [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        r, g, b = pixels[x, y]
        RC[y][x] = r
        GC[y][x] = g
        BC[y][x] = b

# Grayscale channel views (single-band "L")
RC_gray, GC_gray, BC_gray = pil_from_2d(RC), pil_from_2d(GC), pil_from_2d(BC)
RC_gray.save("RC_gray.jpg", format="JPEG")
GC_gray.save("GC_gray.jpg", format="JPEG")
BC_gray.save("BC_gray.jpg", format="JPEG")

show_img(RC_gray, "Red Channel (grayscale RC)")
show_img(GC_gray, "Green Channel (grayscale GC)")
show_img(BC_gray, "Blue Channel (grayscale BC)")

# True-color versions by merging one band with two zero bands
zero = Image.new("L", (W, H), 0)
RC_rgb = Image.merge("RGB", (RC_gray, zero,     zero))
GC_rgb = Image.merge("RGB", (zero,     GC_gray, zero))
BC_rgb = Image.merge("RGB", (zero,     zero,    BC_gray))
RC_rgb.save("RC_rgb.jpg", format="JPEG")
GC_rgb.save("GC_rgb.jpg", format="JPEG")
BC_rgb.save("BC_rgb.jpg", format="JPEG")

show_img(RC_rgb, "Red Channel (true RGB)")
show_img(GC_rgb, "Green Channel (true RGB)")
show_img(BC_rgb, "Blue Channel (true RGB)")

# -----------------------------------------------------------------------------
# 3) GRAYSCALE (AG) BY AVERAGE
# Compute AG(x,y) = floor((R + G + B)/3) via explicit pixel access.
# -----------------------------------------------------------------------------
AG = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        r, g, b = pixels[x, y]
        AG[y][x] = (r + g + b) // 3

AG_img = pil_gray_from_2d(AG)
show_img(AG_img, "Grayscale Image AG")
AG_img.save("AG.jpg", format="JPEG")

# -----------------------------------------------------------------------------
# 4) HISTOGRAMS (RC, GC, BC, AG)
# By spec, count occurrences of values 0..255 by scanning each pixel.
# -----------------------------------------------------------------------------
def histogram_0_255(arr2d):
    hist = [0]*256
    for row in arr2d:
        for v in row:
            hist[v] += 1
    return hist

for name, arr in [("RC", RC), ("GC", GC), ("BC", BC), ("AG", AG)]:
    hist = histogram_0_255(arr)
    plt.plot(range(256), hist)
    plt.title(f"Histogram of {name}")
    plt.xlabel("Intensity (0–255)")
    plt.ylabel("Count")
    plt.xlim(0, 255)
    plt.show()

# -----------------------------------------------------------------------------
# 5) BINARIZATION (AB)
# Prompt for TB; pixels >= TB become 255 else 0. Output is a binary image.
# -----------------------------------------------------------------------------
try:
    TB = int(input("Enter threshold TB (0–255): "))
    if not (0 <= TB <= 255):
        raise ValueError
except ValueError:
    TB = 128
    print("Invalid TB; defaulting to 128.")

AB = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        AB[y][x] = 255 if AG[y][x] >= TB else 0

AB_img = pil_gray_from_2d(AB)
show_img(AB_img, f"Binary Image AB (TB={TB})")
AB_img.save("AB.jpg", format="JPEG")

# -----------------------------------------------------------------------------
# 6) SIMPLE EDGE DETECTION (AE)
# Forward-difference gradients:
#   Gx(x,y) = AG(x+1,y) - AG(x,y)   (last column set to 0)
#   Gy(x,y) = AG(x,y+1) - AG(x,y)   (last row set to 0)
# Gradient magnitude GM = sqrt(Gx^2 + Gy^2); threshold with TE.
# -----------------------------------------------------------------------------
try:
    TE = float(input("Enter edge threshold TE (e.g., 15): "))
except ValueError:
    TE = 20.0
    print("Invalid TE; defaulting to 20.0.")

Gx, Gy = [[0]*W for _ in range(H)], [[0]*W for _ in range(H)]
GM = [[0.0]*W for _ in range(H)]

# Forward differences with boundary handling per spec
for y in range(H):
    for x in range(W-1):
        Gx[y][x] = AG[y][x+1] - AG[y][x]
for y in range(H-1):
    for x in range(W):
        Gy[y][x] = AG[y+1][x] - AG[y][x]

# Gradient magnitude and thresholding
AE = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        gx = float(Gx[y][x])
        gy = float(Gy[y][x])
        gm = (gx*gx + gy*gy) ** 0.5
        AE[y][x] = 255 if gm > TE else 0

AE_img = pil_gray_from_2d(AE)
show_img(AE_img, f"Edge Image AE (TE={TE})")
AE_img.save("AE.jpg", format="JPEG")

# -----------------------------------------------------------------------------
# 7) IMAGE PYRAMID (AG2, AG4, AG8)
# Each level halves the resolution by averaging non-overlapping 2×2 blocks.
# -----------------------------------------------------------------------------
def downsample_2x2(arr2d):
    h, w = len(arr2d), len(arr2d[0])
    H2, W2 = h // 2, w // 2
    out = [[0]*W2 for _ in range(H2)]
    for y2 in range(H2):
        for x2 in range(W2):
            y, x = 2*y2, 2*x2
            s = arr2d[y][x] + arr2d[y][x+1] + arr2d[y+1][x] + arr2d[y+1][x+1]
            out[y2][x2] = s // 4
    return out

AG2 = downsample_2x2(AG)
AG4 = downsample_2x2(AG2)
AG8 = downsample_2x2(AG4)

AG2_img, AG4_img, AG8_img = pil_gray_from_2d(AG2), pil_gray_from_2d(AG4), pil_gray_from_2d(AG8)
show_img(AG2_img, "Image Pyramid: AG2 (256×256)")
show_img(AG4_img, "Image Pyramid: AG4 (128×128)")
show_img(AG8_img, "Image Pyramid: AG8 (64×64)")

AG2_img.save("AG2.jpg", format="JPEG")
AG4_img.save("AG4.jpg", format="JPEG")
AG8_img.save("AG8.jpg", format="JPEG")

