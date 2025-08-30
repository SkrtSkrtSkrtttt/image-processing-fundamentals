# Naafiul Hossain — ESE 358/568 Project 1 (pixel-scanned implementation)
# Requirements: pillow, matplotlib
# Run: python proj1.py (place shed1-small.jpg next to this file)

from PIL import Image
import matplotlib.pyplot as plt

IMAGE_PATH = r"C:\Users\Naafiul Hossain\Downloads\shed1-small (1).jpg"

def load_rgb_u8(path):
    img = Image.open(path).convert("RGB")
    if img.size != (512, 512):
        raise ValueError(f"Image must be 512x512, got {img.size}. Use the provided shed1-small.jpg/shed2-small.jpg.")
    return img  # PIL Image RGB, 8-bit

def show_img(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def pil_gray_from_2d(arr2d):
    # arr2d is a list of lists of 0..255 ints; convert back to PIL L
    out = Image.new("L", (len(arr2d[0]), len(arr2d)))
    out.putdata([v for row in arr2d for v in row])
    return out

def pil_from_2d(arr2d, mode="L"):
    out = Image.new(mode, (len(arr2d[0]), len(arr2d)))
    out.putdata([v for row in arr2d for v in row])
    return out

def save_gray(path, arr2d):
    pil_gray_from_2d(arr2d).save(path, format="JPEG")

# 1) Read and display
A = load_rgb_u8(IMAGE_PATH)
show_img(A, "Original Image A (512×512)")

# Convert to per-pixel list for scanning
W, H = A.size
pixels = A.load()  # direct pixel access (x,y) -> (R,G,B)

# -----------------------------
# 2) Isolate RGB channels
#    (produce grayscale AND true-color versions)
# -----------------------------
# RC, GC, BC already computed by pixel scan below; keep this loop
RC = [[0]*W for _ in range(H)]
GC = [[0]*W for _ in range(H)]
BC = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        r, g, b = pixels[x, y]
        RC[y][x] = r
        GC[y][x] = g
        BC[y][x] = b

# ----- Grayscale (single-band) versions for plots/histograms -----
RC_gray = pil_from_2d(RC)  # mode "L"
GC_gray = pil_from_2d(GC)
BC_gray = pil_from_2d(BC)

RC_gray.save("RC_gray.jpg", format="JPEG")
GC_gray.save("GC_gray.jpg", format="JPEG")
BC_gray.save("BC_gray.jpg", format="JPEG")

plt.imshow(RC_gray, cmap="Reds", vmin=0, vmax=255); plt.title("Red Channel (grayscale)"); plt.axis("off"); plt.show()
plt.imshow(GC_gray, cmap="Greens", vmin=0, vmax=255); plt.title("Green Channel (grayscale)"); plt.axis("off"); plt.show()
plt.imshow(BC_gray, cmap="Blues", vmin=0, vmax=255); plt.title("Blue Channel (grayscale)"); plt.axis("off"); plt.show()

# ----- True-color views (only one channel kept in RGB) -----
zero = Image.new("L", (W, H), 0)
RC_rgb = Image.merge("RGB", (RC_gray, zero,      zero     ))  # (R,0,0)
GC_rgb = Image.merge("RGB", (zero,     GC_gray,  zero     ))  # (0,G,0)
BC_rgb = Image.merge("RGB", (zero,     zero,     BC_gray  ))  # (0,0,B)

RC_rgb.save("RC_rgb.jpg", format="JPEG")
GC_rgb.save("GC_rgb.jpg", format="JPEG")
BC_rgb.save("BC_rgb.jpg", format="JPEG")

plt.imshow(RC_rgb); plt.title("Red Channel (true RGB)"); plt.axis("off"); plt.show()
plt.imshow(GC_rgb); plt.title("Green Channel (true RGB)"); plt.axis("off"); plt.show()
plt.imshow(BC_rgb); plt.title("Blue Channel (true RGB)"); plt.axis("off"); plt.show()


# 3) Grayscale AG by explicit average of R,G,B
AG = [[0]*W for _ in range(H)]
for y in range(H):
    for x in range(W):
        r, g, b = pixels[x, y]
        AG[y][x] = (r + g + b) // 3  # integer average
show_img(pil_gray_from_2d(AG), "Grayscale AG")

# 4) Histograms of RC, GC, BC, AG by explicit counting
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

# 5) Binarization AB with user TB
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

show_img(pil_gray_from_2d(AB), f"Binary AB (TB={TB})")
save_gray("AB.jpg", AB)

# 6) Simple edge detection AE via forward differences, GM threshold TE
try:
    TE = float(input("Enter edge threshold TE (e.g., 15): "))
except ValueError:
    TE = 20.0
    print("Invalid TE; defaulting to 20.0.")

Gx = [[0]*W for _ in range(H)]
Gy = [[0]*W for _ in range(H)]
GM = [[0.0]*W for _ in range(H)]

# Forward differences; last col/row = 0 per spec
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

show_img(pil_gray_from_2d(AE), f"Edge Image AE (TE={TE})")
save_gray("AE.jpg", AE)

# 7) Image pyramid (AG2, AG4, AG8) via 2x2 average (explicit access)
def downsample_2x2(arr2d):
    h = len(arr2d)
    w = len(arr2d[0])
    # assume even (512 is even). If odd, drop last row/col.
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

show_img(pil_gray_from_2d(AG2), "Image Pyramid: AG2 (256×256)")
show_img(pil_gray_from_2d(AG4), "Image Pyramid: AG4 (128×128)")
show_img(pil_gray_from_2d(AG8), "Image Pyramid: AG8 (64×64)")

# Save pyramid images
pil_gray_from_2d(AG2).save("AG2.jpg", format="JPEG")
pil_gray_from_2d(AG4).save("AG4.jpg", format="JPEG")
pil_gray_from_2d(AG8).save("AG8.jpg", format="JPEG")
