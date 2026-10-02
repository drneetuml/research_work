# Hybrid DCT Compression, Chaos Encryption, and Autoencoder Reconstruction

Reproducibility code for the revised manuscript:

**Hybrid DCT-Based Image Compression and Chaos Encryption with Autoencoder-Assisted Reconstruction Enhancement**

## Pipeline

`grayscale image -> 8x8 DCT/Q50 quantization -> chaotic permutation -> bidirectional feedback diffusion -> decryption -> IDCT -> receiver-side convolutional autoencoder`

The autoencoder is **not** used as a transmitter-side encoder. It is a receiver-side reconstruction/enhancement stage.

## Files

- `main_experiment.py` — complete DIV2K training and 24-image Kodak evaluation, including compression, PSNR/MSE/SSIM, entropy, histogram variance, H/V/D correlation, NPCR/UACI, chi-square, Fourier spectral flatness, and timing.
- `reviewer7_validation.py` — additional key-sensitivity and native-resolution Kodak validation.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes datasets, trained weights, and generated results from Git.

## Data

Prepare these directories:

```text
DIV2K_train_HR/
DIV2K_valid_HR/
kodak/
```

The manuscript protocol uses:
- DIV2K: first 116 training images and first 20 validation images found by the script.
- Training patches: 2 x 64x64 patches per training image = 232 samples.
- Validation patches: 1 x 64x64 patch per validation image = 20 samples.
- Kodak: all 24 images are held out from training/validation/model selection.
- Main Kodak evaluation: grayscale 256x256.
- Native-resolution validation: grayscale 512x768 or 768x512 Kodak images.

## Main experiment

Edit the four path variables near the top of `main_experiment.py`:

```python
DIV2K_TRAIN_DIR = "/path/to/DIV2K_train_HR"
DIV2K_VALID_DIR = "/path/to/DIV2K_valid_HR"
KODAK_DIR = "/path/to/kodak"
OUTPUT_DIR = "/path/to/results"
```

Then run:

```bash
pip install -r requirements.txt
python main_experiment.py
```

The autoencoder protocol is 8 epochs with Adam, learning rate `1e-3` for epochs 1-4 and `3e-4` for epochs 5-8. The receiver-side residual blend is `alpha = 0.025`.

## Reviewer-7 validation

```bash
python reviewer7_validation.py --kodak_dir /path/to/kodak --output_dir results_reviewer7
```

The key-sensitivity test changes only the forward logistic-map initial condition from `0.2718281828` to `0.271828182801` (`delta = 1e-12`) during decryption.

The manuscript reports, over all 24 Kodak images:
- recovered-coefficient mismatch: 99.9573%
- wrong-key PSNR: 5.292 +/- 0.402 dB
- wrong-key SSIM: 0.0033 +/- 0.0008
- native-resolution IDCT PSNR: 33.397 +/- 2.389 dB
- native-resolution IDCT SSIM: 0.9168 +/- 0.0161

## Important interpretation notes

1. The DCT path uses 8-bit grayscale values centered by subtracting 128. `[0,1]` normalization is used only at the autoencoder input.
2. The chi-square/Fourier image-statistical analysis uses the low 8-bit ciphertext visualization defined in the implementation; the encrypted coefficient stream itself remains 16-bit.
3. Chi-square histogram uniformity, Fourier spectral flatness, entropy, and correlation are diagnostic statistics; none alone proves cryptographic security.
4. NPCR/UACI values in this compression-before-encryption pipeline are reported as measured and should not be interpreted as ideal differential-security performance.
5. The current implementation is sequential; the repository does not claim unmeasured 1/2/4/8-thread speedups.

## Reproducibility

For publication, record your exact Python/package versions and hardware in the final repository release. Do not commit DIV2K/Kodak datasets unless their licenses explicitly permit redistribution.
