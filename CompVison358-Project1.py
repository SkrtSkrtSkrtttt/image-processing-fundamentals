# Naafiul Hossain — Project 1: RGB Pixel Operations (pixel-scanned)
git add CompVison358-Project1.py AG.jpg

from PIL import Image
import matplotlib.pyplot as plt

# ---- set your image path here ----
IMAGE_PATH = r"C:\Users\Naafiul Hossain\Downloads\shed1-small (1).jpg"
# Or use a relative file next to this script:
# IMAGE_PATH = "shed1-small.jpg"

# ---------- helpers ----------
def load_rgb_u8(path):
    img = Image.open(path).convert("RGB")  # force RGB, 8-bit per channel
    if img.size != (512, 512):
        raise ValueError(f"Image must be 512x512, got {img.size}. Use shed1-small.jpg/shed2-small.jpg.")
    return img

def show_img(img, title):
    # If grayscale (single-band), force gray colormap (no false-color)
    if isinstance(img, Image.Image) and img.mode == "L":
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def pil_gray_from_2d(arr2d):
    # arr2d: list of lists of 0..255 ints (H x W)
    h = len(arr2d)
    w = len(arr2d[0])
    out = Image.new("L", (w, h))
    out.putdata([v for row in arr2d for v in row])
    return out

def pil_from_2d(arr2d, mode="L"):
    h = len(arr2d)
    w = len(arr2d[0])
    out = Image.new(mode, (w, h))
    out.putdata([v for row in arr2d for v in row])
    return out

def save_gray(path, arr2d):
    pil_gray_from_2d(arr2d).save(path, format="JPEG")

# ---------- 1) read & display ----------
A = load_rgb_u8(IMAGE_PATH)
show_img(A, "Original Image A (512×512)")

W, H = A.size
pixels = A.load()  # random-access (x,y) -> (R,G,B)

# ---------- 2) isolate channels (grayscale + true-color) ----------
RC = [[0]*W for _ in range(H)]
GC = [[0]*W for _ in range(H)]
BC = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        r, g, b = pixels[x, y]
        RC[y][x] = r
        GC[y][x] = g
        BC[y][x] = b

RC_gray = pil_from_2d(RC)  # "L"
GC_gray = pil_from_2d(GC)
BC_gray = pil_from_2d(BC)

RC_gray.save("RC_gray.jpg", format="JPEG")
GC_gray.save("GC_gray.jpg", format="JPEG")
BC_gray.save("BC_gray.jpg", format="JPEG")

# Show grayscale channel images in true gray (no false-color)
show_img(RC_gray, "Red Channel (grayscale RC)")
show_img(GC_gray, "Green Channel (grayscale GC)")
show_img(BC_gray, "Blue Channel (grayscale BC)")

# True-color views (only one channel kept)
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

# ---------- 3) grayscale AG by average ----------
AG = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        r, g, b = pixels[x, y]
        AG[y][x] = (r + g + b) // 3

AG_img = pil_gray_from_2d(AG)
show_img(AG_img, "Grayscale Image AG")
AG_img.save("AG.jpg", format="JPEG")

# ---------- 4) histograms by explicit counting ----------
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

# ---------- 5) binarization ----------
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

# ---------- 6) simple edge detection ----------
try:
    TE = float(input("Enter edge threshold TE (e.g., 15): "))
except ValueError:
    TE = 20.0
    print("Invalid TE; defaulting to 20.0.")

Gx = [[0]*W for _ in range(H)]
Gy = [[0]*W for _ in range(H)]
GM = [[0.0]*W for _ in range(H)]

# Forward differences (last col/row remain 0 by spec)
for y in range(H):
    for x in range(W-1):
        Gx[y][x] = AG[y][x+1] - AG[y][x]
for y in range(H-1):
    for x in range(W):
        Gy[y][x] = AG[y+1][x] - AG[y][x]

for y in range(H):
    for x in range(W):
        gx = float(Gx[y][x])
        gy = float(Gy[y][x])
        GM[y][x] = (gx*gx + gy*gy) ** 0.5

AE = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        AE[y][x] = 255 if GM[y][x] > TE else 0

AE_img = pil_gray_from_2d(AE)
show_img(AE_img, f"Edge Image AE (TE={TE})")
AE_img.save("AE.jpg", format="JPEG")

# ---------- 7) image pyramid ----------
def downsample_2x2(arr2d):
    h = len(arr2d)
    w = len(arr2d[0])
    H2 = h // 2
    W2 = w // 2
    out = [[0]*W2 for _ in range(H2)]
    for y2 in range(H2):
        for x2 in range(W2):
            y = 2*y2
            x = 2*x2
            s = arr2d[y][x] + arr2d[y][x+1] + arr2d[y+1][x] + arr2d[y+1][x+1]
            out[y2][x2] = s // 4
    return out

AG2 = downsample_2x2(AG)
AG4 = downsample_2x2(AG2)
AG8 = downsample_2x2(AG4)

AG2_img = pil_gray_from_2d(AG2)
AG4_img = pil_gray_from_2d(AG4)
AG8_img = pil_gray_from_2d(AG8)

show_img(AG2_img, "Image Pyramid: AG2 (256×256)")
show_img(AG4_img, "Image Pyramid: AG4 (128×128)")
show_img(AG8_img, "Image Pyramid: AG8 (64×64)")

AG2_img.save("AG2.jpg", format="JPEG")
AG4_img.save("AG4.jpg", format="JPEG")
AG8_img.save("AG8.jpg", format="JPEG")
