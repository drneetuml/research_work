# ================================================================
# HYBRID DCT COMPRESSION + CHAOS ENCRYPTION +
# AUTOENCODER-ASSISTED IMAGE RECONSTRUCTION
#
# Corresponds to the revised manuscript implementation
# ================================================================

import os
import glob
import time
import zlib
import random
import numpy as np
import pandas as pd
import cv2

from scipy.fftpack import dct, idct
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================================
# 1. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ================================================================
# 2. PATHS
# ================================================================

# Change these paths according to your Google Drive / local folders.

DIV2K_TRAIN_DIR = "/content/DIV2K_train_HR"
DIV2K_VALID_DIR = "/content/DIV2K_valid_HR"
KODAK_DIR       = "/content/kodak"

OUTPUT_DIR = "/content/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 3. JPEG Q50 LUMINANCE QUANTIZATION MATRIX
# ================================================================

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
], dtype=np.float32)


# ================================================================
# 4. IMAGE LOADING
# ================================================================

def read_gray_image(path, size=(256, 256)):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img.astype(np.uint8)


def get_images(folder):

    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))

    return sorted(files)


# ================================================================
# 5. 2-D DCT / IDCT
# ================================================================

def dct2(block):

    return dct(
        dct(block.T, norm="ortho").T,
        norm="ortho"
    )


def idct2(block):

    return idct(
        idct(block.T, norm="ortho").T,
        norm="ortho"
    )


# ================================================================
# 6. BLOCK DCT + Q50 QUANTIZATION
# ================================================================

def compress_dct(image):

    """
    image: uint8 grayscale image

    Returns signed int16 quantized DCT coefficients.
    """

    h, w = image.shape

    coefficients = np.zeros((h, w), dtype=np.int16)

    centered = image.astype(np.float32) - 128.0

    for i in range(0, h, 8):

        for j in range(0, w, 8):

            block = centered[i:i+8, j:j+8]

            transformed = dct2(block)

            quantized = np.round(
                transformed / Q50
            )

            coefficients[i:i+8, j:j+8] = \
                quantized.astype(np.int16)

    return coefficients


# ================================================================
# 7. INVERSE QUANTIZATION + IDCT
# ================================================================

def decompress_dct(coefficients):

    h, w = coefficients.shape

    reconstructed = np.zeros((h, w), dtype=np.float32)

    for i in range(0, h, 8):

        for j in range(0, w, 8):

            qblock = coefficients[i:i+8, j:j+8].astype(np.float32)

            dequantized = qblock * Q50

            block = idct2(dequantized)

            reconstructed[i:i+8, j:j+8] = block

    reconstructed += 128.0

    reconstructed = np.clip(
        np.round(reconstructed),
        0,
        255
    ).astype(np.uint8)

    return reconstructed


# ================================================================
# 8. LOGISTIC CHAOTIC MAP
# ================================================================

def logistic_sequence(length, x0, r=3.99):

    sequence = np.empty(length, dtype=np.float64)

    x = float(x0)

    for i in range(length):

        x = r * x * (1.0 - x)

        sequence[i] = x

    return sequence


# ================================================================
# 9. CHAOTIC PERMUTATION
# ================================================================

def generate_permutation(length,
                         x0=0.3141592653,
                         r=3.99):

    chaotic = logistic_sequence(length, x0, r)

    permutation = np.argsort(
        chaotic,
        kind="mergesort"
    )

    return permutation


# ================================================================
# 10. GENERATE 16-BIT CHAOTIC KEY STREAM
# ================================================================

def chaotic_keystream(length,
                      x0,
                      r=3.99):

    sequence = logistic_sequence(length, x0, r)

    key = np.floor(
        sequence * 65536.0
    ).astype(np.uint32)

    key = np.bitwise_and(
        key,
        0xFFFF
    ).astype(np.uint16)

    return key


# ================================================================
# 11. BIDIRECTIONAL FEEDBACK ENCRYPTION
# ================================================================

def encrypt_coefficients(
        quantized_coefficients,
        permutation_x0=0.3141592653,
        forward_x0=0.2718281828,
        backward_x0=0.6180339887,
        r=3.99,
        IV1=0xA5C3,
        IV2=0x6D2F):

    shape = quantized_coefficients.shape

    # signed int16 -> unsigned 16-bit representation
    plain = quantized_coefficients.reshape(-1).view(np.uint16)

    n = len(plain)

    # ------------------------------------------------------------
    # Chaotic permutation
    # ------------------------------------------------------------

    permutation = generate_permutation(
        n,
        permutation_x0,
        r
    )

    P = plain[permutation]

    # ------------------------------------------------------------
    # Separate forward/backward chaotic streams
    # ------------------------------------------------------------

    Kf = chaotic_keystream(
        n,
        forward_x0,
        r
    )

    Kb = chaotic_keystream(
        n,
        backward_x0,
        r
    )

    # ------------------------------------------------------------
    # Forward diffusion
    #
    # F[0] = P[0] XOR Kf[0] XOR IV1
    # F[i] = P[i] XOR Kf[i] XOR F[i-1]
    # ------------------------------------------------------------

    F = np.empty(n, dtype=np.uint16)

    F[0] = (
        P[0]
        ^ Kf[0]
        ^ np.uint16(IV1)
    )

    for i in range(1, n):

        F[i] = (
            P[i]
            ^ Kf[i]
            ^ F[i-1]
        )

    # ------------------------------------------------------------
    # Backward diffusion
    #
    # C[N-1] = F[N-1] XOR Kb[N-1] XOR IV2
    # C[i]   = F[i] XOR Kb[i] XOR C[i+1]
    # ------------------------------------------------------------

    C = np.empty(n, dtype=np.uint16)

    C[-1] = (
        F[-1]
        ^ Kb[-1]
        ^ np.uint16(IV2)
    )

    for i in range(n - 2, -1, -1):

        C[i] = (
            F[i]
            ^ Kb[i]
            ^ C[i+1]
        )

    return C.reshape(shape)


# ================================================================
# 12. DECRYPTION
# ================================================================

def decrypt_coefficients(
        cipher,
        permutation_x0=0.3141592653,
        forward_x0=0.2718281828,
        backward_x0=0.6180339887,
        r=3.99,
        IV1=0xA5C3,
        IV2=0x6D2F):

    shape = cipher.shape

    C = cipher.reshape(-1).astype(np.uint16)

    n = len(C)

    Kf = chaotic_keystream(
        n,
        forward_x0,
        r
    )

    Kb = chaotic_keystream(
        n,
        backward_x0,
        r
    )

    # ------------------------------------------------------------
    # Reverse backward diffusion
    # ------------------------------------------------------------

    F = np.empty(n, dtype=np.uint16)

    F[-1] = (
        C[-1]
        ^ Kb[-1]
        ^ np.uint16(IV2)
    )

    for i in range(n - 1):

        F[i] = (
            C[i]
            ^ Kb[i]
            ^ C[i+1]
        )

    # ------------------------------------------------------------
    # Reverse forward diffusion
    # ------------------------------------------------------------

    P = np.empty(n, dtype=np.uint16)

    P[0] = (
        F[0]
        ^ Kf[0]
        ^ np.uint16(IV1)
    )

    for i in range(1, n):

        P[i] = (
            F[i]
            ^ Kf[i]
            ^ F[i-1]
        )

    # ------------------------------------------------------------
    # Inverse permutation
    # ------------------------------------------------------------

    permutation = generate_permutation(
        n,
        permutation_x0,
        r
    )

    original = np.empty(n, dtype=np.uint16)

    original[permutation] = P

    # unsigned representation -> signed int16
    original = original.view(np.int16)

    return original.reshape(shape)


# ================================================================
# 13. CIPHER VISUALIZATION
# ================================================================

def cipher_image(cipher):

    """
    Low 8 bits used only for image-statistical visualization.
    """

    return (
        cipher.astype(np.uint16) & 0x00FF
    ).astype(np.uint8)


# ================================================================
# 14. COMPRESSION SIZE, BPP AND COMPRESSION RATIO
# ================================================================

def compression_metrics(qcoeff):

    # Signed int16 quantized coefficient stream
    raw_coeff_stream = qcoeff.astype(np.int16).tobytes()

    # Reproducible encoded-stream estimate
    compressed = zlib.compress(
        raw_coeff_stream,
        level=9
    )

    compressed_bytes = len(compressed)

    h, w = qcoeff.shape

    bpp = (
        compressed_bytes * 8
    ) / (h * w)

    # Original grayscale image = 1 byte/pixel
    original_bytes = h * w

    compression_ratio = (
        original_bytes /
        compressed_bytes
    )

    size_kb = (
        compressed_bytes /
        1024.0
    )

    return size_kb, bpp, compression_ratio


# ================================================================
# 15. AUTOENCODER
# ================================================================

class ConvAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                16, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        z = self.encoder(x)

        return self.decoder(z)


# ================================================================
# 16. DIV2K DATASET
# ================================================================

class DIV2KDataset(Dataset):

    def __init__(
            self,
            image_paths,
            patches_per_image=2,
            patch_size=64,
            seed=42):

        self.samples = []

        rng = np.random.default_rng(seed)

        for path in image_paths:

            img = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            h, w = img.shape

            if h < patch_size or w < patch_size:
                continue

            for _ in range(patches_per_image):

                y = rng.integers(
                    0,
                    h - patch_size + 1
                )

                x = rng.integers(
                    0,
                    w - patch_size + 1
                )

                original = img[
                    y:y+patch_size,
                    x:x+patch_size
                ]

                qcoeff = compress_dct(original)

                degraded = decompress_dct(qcoeff)

                self.samples.append(
                    (
                        degraded.astype(np.float32) / 255.0,
                        original.astype(np.float32) / 255.0
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        degraded, original = self.samples[index]

        degraded = torch.from_numpy(
            degraded
        ).unsqueeze(0)

        original = torch.from_numpy(
            original
        ).unsqueeze(0)

        return degraded, original


# ================================================================
# 17. PREPARE DIV2K TRAIN / VALIDATION SETS
# ================================================================

train_paths = get_images(DIV2K_TRAIN_DIR)
valid_paths = get_images(DIV2K_VALID_DIR)

# Match reported CPU-practical experiment
train_paths = train_paths[:116]
valid_paths = valid_paths[:20]

train_dataset = DIV2KDataset(
    train_paths,
    patches_per_image=2,
    patch_size=64,
    seed=SEED
)

valid_dataset = DIV2KDataset(
    valid_paths,
    patches_per_image=1,
    patch_size=64,
    seed=SEED
)

print(
    "Training patches:",
    len(train_dataset)
)

print(
    "Validation patches:",
    len(valid_dataset)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ================================================================
# 18. TRAIN AUTOENCODER
# ================================================================

model = ConvAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

best_val_loss = float("inf")

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "div2k_autoencoder.pt"
)

for epoch in range(1, 9):

    # Learning-rate schedule used in experiment
    if epoch == 5:

        for group in optimizer.param_groups:
            group["lr"] = 3e-4

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    model.train()

    train_loss = 0.0

    for degraded, target in train_loader:

        degraded = degraded.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(degraded)

        loss = criterion(
            output,
            target
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item() *
            degraded.size(0)
        )

    train_loss /= len(train_dataset)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for degraded, target in valid_loader:

            degraded = degraded.to(device)
            target = target.to(device)

            output = model(degraded)

            loss = criterion(
                output,
                target
            )

            val_loss += (
                loss.item() *
                degraded.size(0)
            )

    val_loss /= len(valid_dataset)

    print(
        f"Epoch {epoch:02d} | "
        f"Train MSE = {train_loss:.6f} | "
        f"Val MSE = {val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


print(
    "\nBest validation MSE:",
    best_val_loss
)


# ================================================================
# 19. LOAD BEST MODEL
# ================================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ================================================================
# 20. AUTOENCODER RECONSTRUCTION
# ================================================================

def autoencoder_output(image):

    x = (
        torch.from_numpy(
            image.astype(np.float32) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = model(x)

    output = (
        output.squeeze()
        .cpu()
        .numpy()
    )

    return np.clip(
        output * 255.0,
        0,
        255
    )


# ================================================================
# 21. VALIDATION-SELECTED RESIDUAL BLENDING
# ================================================================

ALPHA = 0.025


def ae_assisted_reconstruction(idct_image):

    ae = autoencoder_output(idct_image)

    final = (
        (1.0 - ALPHA) *
        idct_image.astype(np.float32)
        +
        ALPHA * ae
    )

    return np.clip(
        np.round(final),
        0,
        255
    ).astype(np.uint8)


# ================================================================
# 22. RECONSTRUCTION METRICS
# ================================================================

def reconstruction_metrics(original, reconstructed):

    original_f = original.astype(np.float64)

    reconstructed_f = reconstructed.astype(np.float64)

    mse = np.mean(
        (original_f - reconstructed_f) ** 2
    )

    psnr = peak_signal_noise_ratio(
        original,
        reconstructed,
        data_range=255
    )

    ssim = structural_similarity(
        original,
        reconstructed,
        data_range=255
    )

    return mse, psnr, ssim


# ================================================================
# 23. ENTROPY
# ================================================================

def image_entropy(image):

    hist = np.bincount(
        image.flatten(),
        minlength=256
    ).astype(np.float64)

    p = hist / hist.sum()

    p = p[p > 0]

    return -np.sum(
        p * np.log2(p)
    )


# ================================================================
# 24. HISTOGRAM VARIANCE
# ================================================================

def histogram_variance(image):

    hist = np.bincount(
        image.flatten(),
        minlength=256
    ).astype(np.float64)

    return np.var(
        hist,
        ddof=0
    )


# ================================================================
# 25. PIXEL CORRELATION
# ================================================================

def correlation(a, b):

    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return np.corrcoef(a, b)[0, 1]


def directional_correlations(image):

    horizontal = correlation(
        image[:, :-1],
        image[:, 1:]
    )

    vertical = correlation(
        image[:-1, :],
        image[1:, :]
    )

    diagonal = correlation(
        image[:-1, :-1],
        image[1:, 1:]
    )

    return horizontal, vertical, diagonal


# ================================================================
# 26. NPCR AND UACI
# ================================================================

def npcr_uaci(cipher1, cipher2):

    c1 = cipher_image(cipher1)
    c2 = cipher_image(cipher2)

    npcr = (
        np.mean(c1 != c2)
        * 100.0
    )

    uaci = (
        np.mean(
            np.abs(
                c1.astype(np.float64)
                -
                c2.astype(np.float64)
            )
        )
        /
        255.0
        *
        100.0
    )

    return npcr, uaci



# ================================================================
# 26A. CHI-SQUARE TEST AND FOURIER SPECTRUM ANALYSIS
# ================================================================

def chi_square_uniformity(image):
    """
    Pearson chi-square goodness-of-fit test against a uniform
    256-level histogram. Returns chi-square statistic and p-value.
    """
    observed = np.bincount(
        image.flatten(),
        minlength=256
    ).astype(np.float64)

    expected = image.size / 256.0

    chi_square = np.sum(
        (observed - expected) ** 2 / expected
    )

    p_value = chi2_dist.sf(
        chi_square,
        df=255
    )

    return chi_square, p_value


def fourier_spectral_flatness(image):
    """
    Quantitative Fourier-domain flatness of the cipher visualization.
    The image mean is removed to suppress the DC component.
    Values closer to 1 indicate a flatter/noise-like power spectrum.
    """
    x = image.astype(np.float64)
    x = x - np.mean(x)

    power = np.abs(
        np.fft.fft2(x)
    ) ** 2

    power = power.ravel()
    power = power[power > 0]

    eps = np.finfo(np.float64).tiny

    geometric_mean = np.exp(
        np.mean(np.log(power + eps))
    )

    arithmetic_mean = np.mean(
        power + eps
    )

    return geometric_mean / arithmetic_mean


def save_fourier_spectrum(image, output_path, title=""):
    """
    Save log-magnitude centered Fourier spectrum for visual inspection.
    """
    x = image.astype(np.float64)
    x = x - np.mean(x)

    spectrum = np.log1p(
        np.abs(
            np.fft.fftshift(
                np.fft.fft2(x)
            )
        )
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(spectrum, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ================================================================
# 27. TEST COMPLETE KODAK DATASET
# ================================================================

kodak_paths = get_images(KODAK_DIR)

print(
    "\nKodak images found:",
    len(kodak_paths)
)

results = []


for image_index, path in enumerate(kodak_paths, start=1):

    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    original = read_gray_image(
        path,
        (256, 256)
    )

    # ------------------------------------------------------------
    # DCT + quantization timing
    # ------------------------------------------------------------

    t0 = time.perf_counter()

    qcoeff = compress_dct(
        original
    )

    t1 = time.perf_counter()

    dct_ms = (
        t1 - t0
    ) * 1000.0

    # ------------------------------------------------------------
    # Compression metrics
    # ------------------------------------------------------------

    size_kb, bpp, cr = compression_metrics(
        qcoeff
    )

    # ------------------------------------------------------------
    # Encryption timing
    # ------------------------------------------------------------

    t0 = time.perf_counter()

    cipher = encrypt_coefficients(
        qcoeff
    )

    t1 = time.perf_counter()

    encryption_ms = (
        t1 - t0
    ) * 1000.0

    # ------------------------------------------------------------
    # Decryption timing
    # ------------------------------------------------------------

    t0 = time.perf_counter()

    recovered_coeff = decrypt_coefficients(
        cipher
    )

    t1 = time.perf_counter()

    decryption_ms = (
        t1 - t0
    ) * 1000.0

    # Exact coefficient recovery check
    exact_recovery = np.array_equal(
        qcoeff,
        recovered_coeff
    )

    # ------------------------------------------------------------
    # IDCT reconstruction
    # ------------------------------------------------------------

    idct_image = decompress_dct(
        recovered_coeff
    )

    # ------------------------------------------------------------
    # AE-assisted reconstruction timing
    # ------------------------------------------------------------

    t0 = time.perf_counter()

    final_image = ae_assisted_reconstruction(
        idct_image
    )

    t1 = time.perf_counter()

    ae_ms = (
        t1 - t0
    ) * 1000.0

    # ------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------

    mse_idct, psnr_idct, ssim_idct = \
        reconstruction_metrics(
            original,
            idct_image
        )

    mse_final, psnr_final, ssim_final = \
        reconstruction_metrics(
            original,
            final_image
        )

    # ------------------------------------------------------------
    # Cipher statistics
    # ------------------------------------------------------------

    cipher_vis = cipher_image(
        cipher
    )

    entropy = image_entropy(
        cipher_vis
    )

    hist_var = histogram_variance(
        cipher_vis
    )

    corr_h, corr_v, corr_d = \
        directional_correlations(
            cipher_vis
        )

    # ------------------------------------------------------------
    # Reviewer 6: Chi-square and Fourier-spectrum analysis
    # ------------------------------------------------------------

    chi_square, chi_square_p = \
        chi_square_uniformity(
            cipher_vis
        )

    spectral_flatness = \
        fourier_spectral_flatness(
            cipher_vis
        )

    # Save representative Fourier spectra for the first four images
    if image_index <= 4:

        spectrum_path = os.path.join(
            OUTPUT_DIR,
            f"{name}_fourier_spectrum.png"
        )

        save_fourier_spectrum(
            cipher_vis,
            spectrum_path,
            title=f"{name} encrypted Fourier spectrum"
        )

    subtotal_ms = (
        dct_ms
        + encryption_ms
        + decryption_ms
        + ae_ms
    )

    results.append({

        "Image": name,

        "Size_KB": size_kb,

        "BPP": bpp,

        "Compression_Ratio": cr,

        "MSE_IDCT": mse_idct,

        "PSNR_IDCT": psnr_idct,

        "SSIM_IDCT": ssim_idct,

        "MSE_Final": mse_final,

        "PSNR_Final": psnr_final,

        "SSIM_Final": ssim_final,

        "Cipher_Entropy": entropy,

        "Histogram_Variance": hist_var,

        "Correlation_H": corr_h,

        "Correlation_V": corr_v,

        "Correlation_D": corr_d,

        "Chi_Square": chi_square,

        "Chi_Square_p_value": chi_square_p,

        "Fourier_Spectral_Flatness": spectral_flatness,

        "DCT_Quant_ms": dct_ms,

        "Encryption_ms": encryption_ms,

        "Decryption_ms": decryption_ms,

        "AE_ms": ae_ms,

        "Subtotal_ms": subtotal_ms,

        "Exact_Recovery": exact_recovery
    })

    print(
        f"{name}: "
        f"CR={cr:.3f}:1, "
        f"PSNR={psnr_final:.3f} dB, "
        f"SSIM={ssim_final:.4f}, "
        f"Entropy={entropy:.4f}"
    )


# ================================================================
# 28. SAVE MAIN KODAK RESULTS
# ================================================================

results_df = pd.DataFrame(
    results
)

results_file = os.path.join(
    OUTPUT_DIR,
    "kodak_24_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    "\nSaved:",
    results_file
)


# ================================================================
# 29. DIFFERENTIAL ATTACK TEST
#
# 10 perturbations/image × 24 images = 240 trials
# ================================================================

rng = np.random.default_rng(SEED)

differential_results = []


for path in kodak_paths:

    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    original = read_gray_image(
        path,
        (256, 256)
    )

    original_q = compress_dct(
        original
    )

    original_cipher = encrypt_coefficients(
        original_q
    )

    h, w = original.shape

    # deterministic positions
    positions = rng.choice(
        h * w,
        size=10,
        replace=False
    )

    for trial, position in enumerate(
            positions,
            start=1):

        row = position // w
        col = position % w

        perturbed = original.copy()

        # +1 gray-level perturbation modulo 256
        perturbed[row, col] = (
            int(perturbed[row, col]) + 1
        ) % 256

        perturbed_q = compress_dct(
            perturbed
        )

        perturbed_cipher = encrypt_coefficients(
            perturbed_q
        )

        npcr, uaci = npcr_uaci(
            original_cipher,
            perturbed_cipher
        )

        same_quantized_stream = np.array_equal(
            original_q,
            perturbed_q
        )

        differential_results.append({

            "Image": name,

            "Trial": trial,

            "Row": row,

            "Column": col,

            "Quantized_Stream_Identical":
                same_quantized_stream,

            "NPCR_percent": npcr,

            "UACI_percent": uaci
        })



# ================================================================
# 29A. SAVE CHI-SQUARE / FOURIER RESULTS
# ================================================================

security_analysis_df = results_df[[
    "Image",
    "Chi_Square",
    "Chi_Square_p_value",
    "Fourier_Spectral_Flatness"
]].copy()

security_analysis_file = os.path.join(
    OUTPUT_DIR,
    "kodak_chi_square_fourier_results.csv"
)

security_analysis_df.to_csv(
    security_analysis_file,
    index=False
)

print(
    "Saved:",
    security_analysis_file
)


# ================================================================
# 30. SAVE DIFFERENTIAL RESULTS
# ================================================================

diff_df = pd.DataFrame(
    differential_results
)

diff_file = os.path.join(
    OUTPUT_DIR,
    "kodak_240_differential_trials.csv"
)

diff_df.to_csv(
    diff_file,
    index=False
)

print(
    "Saved:",
    diff_file
)


# ================================================================
# 31. SUMMARY STATISTICS
# ================================================================

summary_columns = [

    "Size_KB",
    "BPP",
    "Compression_Ratio",

    "MSE_IDCT",
    "PSNR_IDCT",
    "SSIM_IDCT",

    "MSE_Final",
    "PSNR_Final",
    "SSIM_Final",

    "Cipher_Entropy",
    "Histogram_Variance",

    "Correlation_H",
    "Correlation_V",
    "Correlation_D",

    "Chi_Square",
    "Chi_Square_p_value",
    "Fourier_Spectral_Flatness",

    "DCT_Quant_ms",
    "Encryption_ms",
    "Decryption_ms",
    "AE_ms",
    "Subtotal_ms"
]


summary = pd.DataFrame({

    "Mean":
        results_df[summary_columns].mean(),

    "SD":
        results_df[summary_columns].std(ddof=1),

    "Minimum":
        results_df[summary_columns].min(),

    "Maximum":
        results_df[summary_columns].max()
})


summary_file = os.path.join(
    OUTPUT_DIR,
    "kodak_summary.csv"
)

summary.to_csv(
    summary_file
)


# Differential summary

differential_summary = pd.DataFrame({

    "Mean": [
        diff_df["NPCR_percent"].mean(),
        diff_df["UACI_percent"].mean()
    ],

    "SD": [
        diff_df["NPCR_percent"].std(ddof=1),
        diff_df["UACI_percent"].std(ddof=1)
    ],

    "Minimum": [
        diff_df["NPCR_percent"].min(),
        diff_df["UACI_percent"].min()
    ],

    "Maximum": [
        diff_df["NPCR_percent"].max(),
        diff_df["UACI_percent"].max()
    ]

}, index=[
    "NPCR_percent",
    "UACI_percent"
])


diff_summary_file = os.path.join(
    OUTPUT_DIR,
    "differential_summary.csv"
)

differential_summary.to_csv(
    diff_summary_file
)


# ================================================================
# 32. DISPLAY SUMMARY
# ================================================================

print("\n================ KODAK SUMMARY ================\n")

print(summary)

print("\n============= DIFFERENTIAL SUMMARY ============\n")

print(differential_summary)

print(
    "\nExact coefficient recovery:",
    results_df["Exact_Recovery"].all()
)

print(
    "\nAll experiments completed successfully."
)