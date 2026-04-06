# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import shutil

def cv_imread(file_path):
    """Read image supporting non-ASCII file paths via numpy buffer."""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def cv_imwrite(file_path, img):
    """Write image supporting non-ASCII file paths via encoding to buffer."""
    _, ext = os.path.splitext(file_path)
    cv2.imencode(ext, img)[1].tofile(file_path)

class MirageMultiTrackToolbox:
    """
    Decryption engine for mirage-style steganography.
    Handles extraction via parallel transformation tracks and automated scoring.
    """
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.img_raw = cv_imread(str(self.input_path))
        if self.img_raw is None: 
            raise ValueError("Input file read failure.")
            
        self.out_root = self.input_path.parent / f"{self.input_path.stem}_Master_Decrypted"
        self.engines = ["Standard", "Lab_Adaptive", "Vibrant", "Pure_Gamma"]
        
        for eng in self.engines:
            (self.out_root / f"Engine_{eng}").mkdir(parents=True, exist_ok=True)
        
        # Handle Alpha channel separation
        if len(self.img_raw.shape) == 3 and self.img_raw.shape[2] == 4:
            self.img = self.img_raw[:, :, :3]
            self.alpha = self.img_raw[:, :, 3]
        else:
            self.img = self.img_raw
            self.alpha = None

    # --- Transformation Engines ---

    def _eng_standard(self, img):
        """Track 1: Linear BGR scaling based on 98th percentile intensity."""
        p98 = np.percentile(img, 98)
        scale = min(255.0 / (p98 + 1e-6), 15.0)
        return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    def _eng_lab_adaptive(self, img):
        """Track 2: CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _eng_vibrant(self, img):
        """Track 3: HSV reinforcement for saturation and value recovery."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        v = np.clip((v - np.percentile(v, 1)) * 1.5, 0, 255)
        s = np.clip(s * 1.4, 0, 255)
        return cv2.cvtColor(cv2.merge((h, s, v)).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _eng_pure_gamma(self, img):
        """Track 4: Inverse Gamma transformation (0.45) for deep detail extraction."""
        inv_gamma = 1.0 / 0.45
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def process_and_save(self, img_dark, base_name):
        """Pass input through all defined engines and save outputs."""
        if img_dark is None: return
        for eng in self.engines:
            worker = getattr(self, f"_eng_{eng.lower()}")
            res = worker(img_dark)
            cv_imwrite(str(self.out_root / f"Engine_{eng}" / f"{base_name}.png"), res)

    # --- Extraction Algorithms ---

    def run_all_algos(self):
        """Execute spatial and morphological extraction methods."""
        # Alpha channel extraction
        if self.alpha is not None:
            self.process_and_save(cv2.cvtColor(self.alpha, cv2.COLOR_GRAY2BGR), "00_Alpha")

        # Morphological erosion to isolate dark pixel clusters
        for s in [2, 3]:
            eroded = cv2.erode(self.img, np.ones((s, s), np.uint8))
            self.process_and_save(eroded, f"01_Morph_Erosion_{s}x{s}")

        # Smart Isolation via inpainting mask for high-frequency regions
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        mask = (gray > np.percentile(gray, 35)).astype(np.uint8) * 255
        self.process_and_save(cv2.inpaint(self.img, mask, 1, cv2.INPAINT_NS), "02_Smart_Isolation")

        # Spatial de-interlacing for vertical and checkerboard patterns
        h, w = self.img.shape[:2]
        patterns = {'V': np.fromfunction(lambda y, x: x % 2, (h, w), dtype=int),
                    'C': np.fromfunction(lambda y, x: (x + y) % 2, (h, w), dtype=int)}
        for name, grid in patterns.items():
            for p in [0, 1]:
                mask = (grid == p)
                extracted = np.zeros_like(self.img)
                extracted[mask] = self.img[mask]
                self.process_and_save(extracted, f"03_Spatial_{name}_P{p}")

    def select_best(self):
        """
        Evaluate results based on Shannon entropy and standard deviation.
        Identifies output with highest information density and balanced contrast.
        """
        best_score = -1
        best_path = None
        
        for path in self.out_root.rglob("*.png"):
            img = cv_imread(str(path))
            if img is None: continue
            
            # Entropy calculation for detail richness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist /= (hist.sum() + 1e-7)
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Standard deviation for contrast measurement
            std_dev = np.std(gray)
            
            # Weighted score: Entropy (0.6) + Contrast (0.4)
            score = entropy * 0.6 + (std_dev / 128.0) * 0.4
            
            if score > best_score:
                best_score = score
                best_path = path
        
        if best_path:
            shutil.copy(str(best_path), str(self.out_root / "best.png"))

    @staticmethod
    def encrypt_image(hidden_p, cover_p, out_p):
        """
        Generate mirage image via luminance multiplexing.
        Hidden layer is compressed to [0, 30]; cover layer expanded to [155, 255].
        """
        h_img, c_img = cv_imread(hidden_p), cv_imread(cover_p)
        h, w = c_img.shape[:2]
        h_img = cv2.resize(h_img, (w, h))
        
        h_dark = (h_img.astype(np.float32) * (30.0 / 255.0)).astype(np.uint8)
        c_bright = (c_img.astype(np.float32) * (100.0 / 255.0) + 155.0).astype(np.uint8)
        
        # Checkerboard interlacing mask
        mask = (np.fromfunction(lambda y, x: (x + y) % 2, (h, w), dtype=int) == 1)[..., np.newaxis]
        cv_imwrite(out_p, np.where(mask, c_bright, h_dark))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-m", "--mode", choices=["decrypt", "encrypt"], default="decrypt")
    parser.add_argument("-c", "--cover")
    parser.add_argument("-o", "--output", default="Result.png")
    args = parser.parse_args()

    if args.mode == "decrypt":
        tool = MirageMultiTrackToolbox(args.input)
        tool.run_all_algos()
        tool.select_best()
    else:
        MirageMultiTrackToolbox.encrypt_image(args.input, args.cover, args.output)