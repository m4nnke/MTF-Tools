## MTF Calculator – Slanted Edge Analysis

This repository contains a Python script for analyzing image sharpness using a **slanted edge MTF (Modulation Transfer Function)** method. It is intended as a **practical lab reference tool**, not as a full compliance or certification implementation of any standard.

### What the script does

- **Loads an image** (e.g. a test chart with a slanted edge).
- **Lets you select / define an edge region** to analyze.
- **Computes the MTF** from the edge profile using a standard slanted‑edge workflow.
- **Displays and/or saves plots and key metrics** that help you judge lens / system sharpness.

The implementation is inspired by common **ISO-style slanted edge methods** (such as ISO 12233 concepts), but:

- **It is not a 1:1 implementation of any ISO standard.**
- **It should not be used as a primary certification or reference tool.**
- **It is best seen as a “good lab reference” for relative comparisons and development work.**

### Requirements & installation

1. **Python version**
  - Use a relatively recent **Python 3.x** (e.g. 3.9+ recommended).
2. **Install dependencies**
  - From the repository root, run:

```bash
pip install -r requirements.txt
```

If you work in a virtual environment (recommended), create and activate it before running the command above.

### Basic usage

From the repository root, run the main analysis script:

```bash
python mtf_analysis.py [dir] [options]
```

If you omit `dir`, the script uses the current directory (`.`) and processes all supported image files in it.

Typical workflow (high level):

- **1. Provide an input image**  
  - Use your own image or the examples in `example_img/`.  
  - The image should contain a **clear slanted edge** suitable for MTF analysis.
- **2. Select / configure the edge region**  
  - Depending on how the script is implemented, you may:
    - Edit configuration values in the script, or
    - Use an interactive selection, or
    - Provide command line arguments.
  - In any case, the goal is to ensure the script focuses on the **correct slanted edge area**.
- **3. Run the analysis**  
  - The script will:
    - Extract the edge spread function (ESF).
    - Derive the line spread function (LSF).
    - Compute the MTF curve.
- **4. Inspect results**  
  - Look at:
    - The **MTF curve** (frequency vs. contrast).
    - Summary metrics (e.g., MTF at certain spatial frequencies).
  - Use these results to compare lenses, apertures, focus settings, or processing pipelines.

### Command-line arguments

The script accepts the following arguments:

- `**dir`** (positional, optional)  
  - Input directory containing images to analyze.  
  - **Default**: current directory (`.`).  
  - All images with typical extensions are processed.
- `**--output`, `-o`**  
  - Output directory for plots and results.  
  - **Default**: `output`.
- `**--verbose`**  
  - Enables more detailed logging to the console.
- `**--debug-plots**`  
  - Saves additional debug visualizations:
    - Auto-ROI detection debug image.
    - Edge overlay image for the detected edge.
- `**--lpmm**`  
  - Interprets MTF frequencies in **line pairs per millimeter (lp/mm)** instead of cycles per pixel.  
  - When enabled, the script asks you for the **pixel size in µm** for each image (with a configurable default).
- `**--MTF50`**  
  - Computes and annotates the **MTF50** (frequency where MTF falls to 50%) on the summary MTF plot.
- `**--default-pixel`**  
  - Default pixel size in **micrometers (µm)** used when `--lpmm` is active and you accept the default per-image value.  
  - **Default**: `3.76`.
- `**--manual-roi`**  
  - Disables auto-ROI.  
  - Opens an interactive window so you can manually draw/select the region of interest around the slanted edge for each image.
- `**--esf-lsf-fwhm**`  
  - Generates an additional per-image PDF showing:
    - ESF (Edge Spread Function),
    - LSF (Line Spread Function),
    - The **FWHM** (Full Width at Half Maximum) if it can be estimated.

### Notes and limitations

- **Standard method, but not a standard reference**  
  - The approach follows a **standard slanted edge method inspired by ISO-style procedures**.  
  - Implementation details and simplifications mean this script **does not replace a fully validated ISO 12233 tool**.
- **Intended use**  
  - **Great for**: development work, relative lens/system comparisons, lab experimentation.  
  - **Not intended for**: official certification, contractual image quality guarantees, or regulatory submissions.
- **Accuracy considerations**  
  - Results will depend on:
    - Input image quality (noise, exposure, chart quality).
    - Correct edge selection and orientation.
    - Sampling, interpolation, and processing parameters inside the script.

### Getting help or extending the script

- If you are familiar with Python and image processing, you can:
  - Adapt the slanted edge workflow (e.g. different frequency units, binning, or windowing).
  - Integrate the script into a larger lab pipeline.
  - Add your own plotting or reporting.

Since the script aims to stay **practical rather than “standard-perfect”**, feel free to tailor it to your specific lab setup and image quality questions.