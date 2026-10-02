"""
Additional validation experiments for the revised manuscript:
1) Key sensitivity on all 24 Kodak images.
2) Native-resolution DCT/encryption/decryption/IDCT scalability check.

This script uses the SAME Q50 DCT and chaotic permutation/bidirectional
feedback-diffusion definitions as main_experiment.py.

Usage:
    python reviewer7_validation.py --kodak_dir /path/to/kodak --output_dir results_r7
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

Q50 = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

KEY = dict(
    permutation_x0=0.3141592653,
    forward_x0=0.2718281828,
    backward_x0=0.6180339887,
    r=3.99,
    iv1=0xA5C3,
    iv2=0x6D2F,
)

def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def idct2(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")

def compress_dct(image):
    h, w = image.shape
    if h % 8 or w % 8:
        raise ValueError("Image dimensions must be divisible by 8.")
    out = np.zeros((h, w), dtype=np.int16)
    x = image.astype(np.float32) - 128.0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            out[i:i+8,j:j+8] = np.round(dct2(x[i:i+8,j:j+8]) / Q50).astype(np.int16)
    return out

def decompress_dct(coeff):
    h, w = coeff.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            out[i:i+8,j:j+8] = idct2(coeff[i:i+8,j:j+8].astype(np.float32) * Q50)
    return np.clip(np.round(out + 128.0), 0, 255).astype(np.uint8)

def logistic_sequence(length, x0, r=3.99):
    seq = np.empty(length, dtype=np.float64)
    x = float(x0)
    for i in range(length):
        x = r * x * (1.0 - x)
        seq[i] = x
    return seq

def generate_permutation(length, x0, r=3.99):
    return np.argsort(logistic_sequence(length, x0, r), kind="mergesort")

def chaotic_keystream(length, x0, r=3.99):
    return (np.floor(logistic_sequence(length, x0, r) * 65536)
            .astype(np.uint32) & 0xFFFF).astype(np.uint16)

def encrypt_coefficients(q, permutation_x0=KEY["permutation_x0"],
                         forward_x0=KEY["forward_x0"],
                         backward_x0=KEY["backward_x0"],
                         r=KEY["r"], iv1=KEY["iv1"], iv2=KEY["iv2"]):
    p = q.reshape(-1).view(np.uint16)
    n = len(p)
    perm = generate_permutation(n, permutation_x0, r)
    P = p[perm]
    kf = chaotic_keystream(n, forward_x0, r)
    kb = chaotic_keystream(n, backward_x0, r)
    F = np.empty(n, dtype=np.uint16)
    F[0] = P[0] ^ kf[0] ^ np.uint16(iv1)
    for i in range(1, n):
        F[i] = P[i] ^ kf[i] ^ F[i-1]
    C = np.empty(n, dtype=np.uint16)
    C[-1] = F[-1] ^ kb[-1] ^ np.uint16(iv2)
    for i in range(n-2, -1, -1):
        C[i] = F[i] ^ kb[i] ^ C[i+1]
    return C.reshape(q.shape)

def decrypt_coefficients(C, permutation_x0=KEY["permutation_x0"],
                         forward_x0=KEY["forward_x0"],
                         backward_x0=KEY["backward_x0"],
                         r=KEY["r"], iv1=KEY["iv1"], iv2=KEY["iv2"]):
    shape = C.shape
    C = C.reshape(-1).astype(np.uint16)
    n = len(C)
    kf = chaotic_keystream(n, forward_x0, r)
    kb = chaotic_keystream(n, backward_x0, r)
    F = np.empty(n, dtype=np.uint16)
    F[-1] = C[-1] ^ kb[-1] ^ np.uint16(iv2)
    for i in range(n-1):
        F[i] = C[i] ^ kb[i] ^ C[i+1]
    P = np.empty(n, dtype=np.uint16)
    P[0] = F[0] ^ kf[0] ^ np.uint16(iv1)
    for i in range(1, n):
        P[i] = F[i] ^ kf[i] ^ F[i-1]
    original = np.empty(n, dtype=np.uint16)
    original[generate_permutation(n, permutation_x0, r)] = P
    return original.view(np.int16).reshape(shape)

def image_paths(folder):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)

def run_key_sensitivity(paths, output_dir, delta=1e-12):
    rows = []
    for path in paths:
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        original = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA)
        q = compress_dct(original)
        cipher = encrypt_coefficients(q)
        wrong_q = decrypt_coefficients(cipher, forward_x0=KEY["forward_x0"] + delta)
        wrong_img = decompress_dct(wrong_q)
        rows.append({
            "Image": os.path.basename(path),
            "Delta_forward_x0": delta,
            "Coefficient_mismatch_percent": 100.0*np.mean(q.reshape(-1) != wrong_q.reshape(-1)),
            "Wrong_key_PSNR_dB": peak_signal_noise_ratio(original, wrong_img, data_range=255),
            "Wrong_key_SSIM": structural_similarity(original, wrong_img, data_range=255),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "key_sensitivity_results.csv"), index=False)
    return df

def run_native_resolution(paths, output_dir):
    rows = []
    for path in paths:
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = original.shape
        q = compress_dct(original)
        cipher = encrypt_coefficients(q)
        recovered_q = decrypt_coefficients(cipher)
        reconstructed = decompress_dct(recovered_q)
        rows.append({
            "Image": os.path.basename(path),
            "Height": h,
            "Width": w,
            "Exact_coeff_recovery": bool(np.array_equal(q, recovered_q)),
            "PSNR_IDCT_dB": peak_signal_noise_ratio(original, reconstructed, data_range=255),
            "SSIM_IDCT": structural_similarity(original, reconstructed, data_range=255),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "native_resolution_results.csv"), index=False)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kodak_dir", required=True)
    parser.add_argument("--output_dir", default="results_reviewer7")
    parser.add_argument("--key_delta", type=float, default=1e-12)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    paths = image_paths(args.kodak_dir)
    if len(paths) != 24:
        print(f"Warning: expected 24 Kodak images; found {len(paths)}.")
    ks = run_key_sensitivity(paths, args.output_dir, args.key_delta)
    nr = run_native_resolution(paths, args.output_dir)

    print("\nKey sensitivity summary")
    print(ks[["Coefficient_mismatch_percent","Wrong_key_PSNR_dB","Wrong_key_SSIM"]]
          .agg(["mean","std","min","max"]))
    print("\nNative-resolution summary")
    print(nr[["PSNR_IDCT_dB","SSIM_IDCT"]].agg(["mean","std","min","max"]))
    print("\nExact coefficient recovery for all images:", nr["Exact_coeff_recovery"].all())

if __name__ == "__main__":
    main()
