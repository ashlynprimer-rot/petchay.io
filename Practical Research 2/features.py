import cv2
import numpy as np
from skimage import measure

def load_image_bytes(file_bytes):
    # Return OpenCV BGR image and RGB PIL if needed
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    return img

def segment_leaf_bgr(img_bgr):
    """
    Simple segmentation:
     - convert to HSV, threshold by green-ish hue ranges
     - fallback to Otsu on green channel if necessary
    Returns mask (uint8 0/255) and masked image.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # green hue range in degrees mapped to 0..179: approx 35..85 => 17..42 on OpenCV scale
    lower = np.array([35//2, 40, 20])   # 17, 40, 20
    upper = np.array([85//2, 255, 255]) # 42, 255, 255
    mask = cv2.inRange(img_hsv, lower, upper)

    # Morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # If mask is tiny (fail), use Otsu on green channel
    if mask.sum() < 50:
        g = img_bgr[:,:,1]
        _, mask2 = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask = mask2

    # Keep largest connected component (assume main plant)
    labels = measure.label(mask > 0)
    if labels.max() == 0:
        return mask, img_bgr
    props = measure.regionprops(labels)
    # find largest by area
    largest = max(props, key=lambda p: p.area)
    largest_mask = (labels == largest.label).astype(np.uint8) * 255

    return largest_mask.astype(np.uint8), cv2.bitwise_and(img_bgr, img_bgr, mask=largest_mask.astype(np.uint8))

def compute_color_indices(img_bgr, mask):
    """Compute average RGB, ExG, VARI, GLI on masked pixels."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    masked = mask > 0
    if masked.sum() == 0:
        return {}
    R = img_rgb[:,:,0][masked]
    G = img_rgb[:,:,1][masked]
    B = img_rgb[:,:,2][masked]

    mean_R = float(R.mean())
    mean_G = float(G.mean())
    mean_B = float(B.mean())

    # Extra green (ExG) = 2G - R - B
    exg = 2*G - R - B
    mean_exg = float(np.mean(exg))

    # VARI = (G - R) / (G + R - B)   (avoid div0)
    denom = (G + R - B)
    denom[denom == 0] = 1e-6
    vari = (G - R) / denom
    mean_vari = float(np.mean(vari))

    # GLI (Green Leaf Index) = (2G - R - B)/(2G + R + B)
    denom2 = (2*G + R + B)
    denom2[denom2 == 0] = 1e-6
    gli = (2*G - R - B)/denom2
    mean_gli = float(np.mean(gli))

    return {
        "mean_R": mean_R,
        "mean_G": mean_G,
        "mean_B": mean_B,
        "mean_exg": mean_exg,
        "mean_vari": mean_vari,
        "mean_gli": mean_gli
    }

def shape_and_texture_features(mask):
    """Compute area, perimeter, aspect ratio, solidity."""
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return {}
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h if h > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity
    }

def texture_from_image(img_bgr, mask):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = gray.copy()
    masked_gray[mask==0] = 0
    # Laplacian variance
    lap_var = float(cv2.Laplacian(masked_gray, cv2.CV_64F).var())
    # entropy: compute histogram on masked pixels
    vals = masked_gray[mask>0]
    if vals.size == 0:
        ent = 0.0
    else:
        hist = np.bincount(vals, minlength=256).astype(np.float32)
        hist = hist / hist.sum()
        hist_nonzero = hist[hist>0]
        ent = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))
    return {"lap_var": lap_var, "entropy": ent}

def extract_features_from_bytes(file_bytes):
    img_bgr = load_image_bytes(file_bytes)
    mask, masked_img = segment_leaf_bgr(img_bgr)
    color_feats = compute_color_indices(img_bgr, mask)
    shape_feats = shape_and_texture_features(mask)
    text_feats = texture_from_image(img_bgr, mask)
    # Combine
    features = {}
    features.update(color_feats)
    features.update(shape_feats)
    features.update(text_feats)
    # add pixel_area_to_cm2 if you have reference (not included)
    return features

# quick test usage:
if __name__ == "__main__":
    with open("example_pechay.jpg","rb") as f:
        b = f.read()
    feats = extract_features_from_bytes(b)
    print(feats)