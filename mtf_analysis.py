import sys
import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from skimage import io, color, feature, transform, filters
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

matplotlib.use('Qt5Agg')

@dataclass
class AnalysisConfig:
    """Configuration options that control how the MTF analysis is executed."""
    input_dir: Path
    output_dir: Path
    verbose: bool = False
    use_lpmm: bool = False
    default_pixel_size: float = 3.76
    show_debug_plots: bool = True
    mtf50: bool = False
    manual_roi: bool = False
    esf_lsf_fwhm: bool = False

    def setup_logging(self):
        """Configure the global logging level and silence noisy third‑party loggers."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

@dataclass
class AutoRoiDebugInfo:
    """Intermediate data used to visualize the automatic ROI selection."""
    full_image: np.ndarray
    grad_image: np.ndarray
    edge_x_coords: np.ndarray
    edge_y_coords: np.ndarray
    center_x: int
    center_y: int
    roi_coords: Tuple[int, int, int, int] # x1, x2, y1, y2

@dataclass
class EdgeDebugInfo:
    """Information about the detected edge inside the ROI."""
    roi_image: np.ndarray
    edge_angle: float
    edge_dist: float

@dataclass
class MtfResult:
    """Container for all intermediate and final MTF results for a single image."""
    name: str
    esf_raw: np.ndarray
    lsf_raw: np.ndarray
    mtf_raw: np.ndarray
    freqs_raw: np.ndarray
    fwhm_px: Optional[float]
    pixel_size_used: Optional[float]
    edge_debug_info: Optional[EdgeDebugInfo] = None
    auto_roi_debug: Optional[AutoRoiDebugInfo] = None

class MtfProcessingPipeline:
    def __init__(self, config: AnalysisConfig):
        """Create a new processing pipeline with the given configuration."""
        self.config = config

    def run(self, image_path: Path, pixel_size_um: float = None) -> Optional[MtfResult]:
        """Run the full MTF processing pipeline on a single image.

        Parameters
        ----------
        image_path:
            Path to the image file to be processed.
        pixel_size_um:
            Pixel pitch in micrometers. Required when `use_lpmm` is enabled.

        Returns
        -------
        MtfResult or None
            The computed MTF result, or ``None`` if processing failed.
        """
        try:
            # 1. Load & Linearize
            img_gray = self._load_linear_image(image_path)
            
            # 2. ROI Selection
            if self.config.manual_roi:
                roi_coords = self._select_roi(img_gray, image_path.name)
            else:
                roi_coords, roi_debug_data = self._auto_select_roi(img_gray, image_path.name, 300)


            img_roi = img_gray[roi_coords[2]:roi_coords[3], roi_coords[0]:roi_coords[1]]

            if img_roi.size == 0:
                raise ValueError("Selected ROI is empty.")

            # 3. Edge Detection
            angle, dist = self._detect_edge_geometry(img_roi, image_path.stem)
            logging.info(f"[{image_path.name}] Edge Angle: {np.rad2deg(angle):.2f}°")

            # Ensure Slanted Edge (Auto-Rotate if necessary)
            img_roi, angle, dist = self._ensure_slanted_edge(img_roi, angle, dist, image_path.stem)

            logging.info(f"[{image_path.name}] Final Edge Angle: {np.rad2deg(angle):.2f}°")

            # 4. Compute ESF
            esf = self._compute_oversampled_esf(img_roi, angle, dist)
            
            # Check for bad data (Zeros in the middle of ESF)
            if np.any(esf == 0):
                logging.warning(f"[{image_path.name}] ESF contains ZEROS! This implies black pixels in the ROI or bad edge detection. Check '_debug_edge_overlay.png'.")

            # 5. Compute LSF
            lsf = np.gradient(esf)

            # 6. Compute MTF
            freqs, mtf = self._compute_mtf(lsf, pixel_size_um)

            # 7. Compute FWHM
            fwhm = self._calculate_fwhm(lsf)

            # Pack edge debug info
            edge_debug_info = EdgeDebugInfo(roi_image=img_roi, edge_angle=angle, edge_dist=dist)

            return MtfResult(
                name=image_path.stem,
                esf_raw=esf,
                lsf_raw=lsf,
                mtf_raw=mtf,
                freqs_raw=freqs,
                fwhm_px=fwhm,
                pixel_size_used=pixel_size_um,
                edge_debug_info=edge_debug_info,
                auto_roi_debug=roi_debug_data
            )

        except Exception as e:
            logging.error(f"Processing failed for {image_path.name}: {e}")
            return None

    def _ensure_slanted_edge(self, img_roi: np.ndarray, angle: float, dist: float, base_name: str) -> Tuple[np.ndarray, float, float]:
        """Ensure the detected edge is slightly slanted; auto‑rotate the ROI if necessary."""
        degrees = np.rad2deg(angle)
        
        is_vertical = abs(degrees) < 2.0 or abs(abs(degrees) - 180.0) < 2.0
        is_horizontal = abs(abs(degrees) - 90.0) < 2.0
        
        if is_vertical or is_horizontal:
            logging.warning(f"[{base_name}] Edge is too straight ({degrees:.2f}°). Auto-rotating ROI by 5° to force a slant.")
            
            try:
                img_rot = self._rotate_and_crop(img_roi, -5.0)
            
                angle_rot, dist_rot = self._detect_edge_geometry(img_rot, f"{base_name}_rotated")

                return img_rot, angle_rot, dist_rot
            except Exception as e:
                logging.warning(f"[{base_name}] Rotation failed to produce a clean edge. Falling back to straight processing.")
                
        return img_roi, angle, dist

    def _rotate_and_crop(self, img: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Rotate an image and crop away the black borders introduced by rotation."""
        rotated = transform.rotate(img, angle_degrees, resize=False, order=3)
        
        h, w = img.shape[:2]
        ang_rad = np.radians(abs(angle_degrees))
        
        crop_x = int(np.ceil(h * np.sin(ang_rad)))
        crop_y = int(np.ceil(w * np.sin(ang_rad)))
        
        crop_x += 2
        crop_y += 2
        
        if crop_x >= w // 2 or crop_y >= h // 2:
            raise ValueError(f"Rotation angle ({angle_degrees}°) is too large; crop would destroy ROI.")
            
        return rotated[crop_y : h - crop_y, crop_x : w - crop_x]

    def _load_linear_image(self, path: Path) -> np.ndarray:
        """Load an image file and return a linearized luminance channel in ``[0, 1]``."""
        img = io.imread(str(path))
        
        if img.ndim == 3 and img.shape[2] == 4:
            img = color.rgba2rgb(img)

        img_f = img.astype(np.float32)
        if img.dtype == np.uint16:
            img_f /= 65535.0
        elif img_f.max() > 1.0:
            img_f /= 255.0

        mask = img_f <= 0.04045
        img_lin = np.empty_like(img_f)
        img_lin[mask] = img_f[mask] / 12.92
        img_lin[~mask] = ((img_f[~mask] + 0.055) / 1.055) ** 2.4

        if img_lin.ndim == 3:
            return 0.2126 * img_lin[..., 0] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 2]
        return img_lin
    
    def _select_roi(self, img: np.ndarray, title: str) -> Tuple[int, int, int, int]:
        """Let the user manually select an ROI via an interactive rectangle selector."""
        print(f"Please select ROI for {title}...")
        roi = []
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Select ROI for {title}")
        
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            roi.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))
            plt.close(fig)

        _ = RectangleSelector(ax, onselect, useblit=True, button=[1], 
                              minspanx=5, interactive=True)
        plt.show()
        if not roi: raise ValueError("ROI not selected")

        logging.info(f"[{title}] Manual-ROI set at (x:{roi[-1][0]}-{roi[-1][1]}, y:{roi[-1][2]}-{roi[-1][3]})")

        return roi[-1]

    def _auto_select_roi(self, img: np.ndarray, title: str, roi_size: int = 400) -> Tuple[int, int, int, int]:
        """Automatically find a sharp edge and center a square ROI around it."""
        logging.info(f"[{title}] Auto-detecting ROI...")
        
        grad = filters.sobel(img)
        
        thresh = np.percentile(grad, 99.5)
        strong_edges = grad > thresh
        
        y_coords, x_coords = np.where(strong_edges)
        
        if len(y_coords) == 0:
            raise ValueError("Auto-ROI failed: Could not detect any sharp edges.")
            
        center_y = int(np.median(y_coords))
        center_x = int(np.median(x_coords))
        
        h, w = img.shape
        half = roi_size // 2
        
        x1 = max(0, center_x - half)
        y1 = max(0, center_y - half)
        x2 = min(w, center_x + half)
        y2 = min(h, center_y + half)
        roi_coords = (x1, x2, y1, y2)
        
        logging.info(f"[{title}] Auto-ROI found at (x:{x1}-{x2}, y:{y1}-{y2})")

        debug_info = AutoRoiDebugInfo(
            full_image=img, grad_image=grad, 
            edge_x_coords=x_coords, edge_y_coords=y_coords,
            center_x=center_x, center_y=center_y, roi_coords=roi_coords
        )

        return roi_coords, debug_info

    def _detect_edge_geometry(self, img: np.ndarray, debug_name: str) -> Tuple[float, float]:
        """Estimate edge angle and distance from the origin inside the ROI using a Hough transform."""
        edges = feature.canny(img, sigma=2)
        hspace, angles, dists = transform.hough_line(edges)
        _, angle_peaks, dist_peaks = transform.hough_line_peaks(hspace, angles, dists, num_peaks=1)
        
        if len(angle_peaks) == 0: 
            raise ValueError("No edge found in ROI")
        
        angle = angle_peaks[0]
        dist = dist_peaks[0]

        return angle, dist

    def _compute_oversampled_esf(self, img: np.ndarray, angle: float, dist: float) -> np.ndarray:
        """Compute an oversampled edge spread function (ESF) along the detected edge."""
        yy, xx = np.indices(img.shape)
        distances = xx * np.cos(angle) + yy * np.sin(angle) - dist
        
        bin_width = 0.25
        bins = np.arange(distances.min(), distances.max(), bin_width)
        
        if len(bins) < 2: return np.zeros(1)

        indices = np.digitize(distances.ravel(), bins) - 1
        valid_idx = (indices >= 0) & (indices < len(bins))
        indices = indices[valid_idx]
        pixel_vals = img.ravel()[valid_idx]

        counts = np.bincount(indices, minlength=len(bins))
        sums = np.bincount(indices, weights=pixel_vals, minlength=len(bins))
        
        valid = counts > 0
        esf = np.zeros(len(bins), dtype=np.float32)
        esf[valid] = sums[valid] / counts[valid]
        
        if np.any(~valid):
            x_idx = np.arange(len(bins))
            esf[~valid] = np.interp(x_idx[~valid], x_idx[valid], esf[valid])

        if np.sum(esf[len(esf)//2:]) < np.sum(esf[:len(esf)//2]):
            esf = np.flip(esf)
            
        return esf

    def _compute_mtf(self, lsf: np.ndarray, pixel_size_um: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the MTF curve from the LSF using a windowed FFT.

        Returns spatial frequencies and MTF values in percent.
        """
        window = np.hanning(len(lsf))
        lsf_w = lsf * window
        
        mtf = np.abs(fftshift(fft(lsf_w)))
        freqs = fftshift(fftfreq(len(lsf), d=0.25))
        
        mid = len(freqs)//2
        mtf = mtf[mid:]
        freqs = freqs[mid:]
        
        if mtf[0] > 0: mtf /= mtf[0]
        
        if self.config.use_lpmm and pixel_size_um is not None:
            freqs /= (pixel_size_um / 1000.0)
            
        return freqs, mtf * 100

    def _calculate_fwhm(self, lsf: np.ndarray) -> Optional[float]:
        """Calculate the full width at half maximum (FWHM) of the LSF in pixels."""
        if lsf.max() == 0: return None
        half_max = lsf.max() / 2
        above = lsf >= half_max
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]
        
        if len(crossings) >= 2:
            x_vals = []
            for idx in [crossings[0], crossings[-1]]:
                y0, y1 = lsf[idx], lsf[idx + 1]
                slope = y1 - y0
                x_cross = idx + (half_max - y0) / slope if slope != 0 else idx
                x_vals.append(x_cross)
            
            return (x_vals[1] - x_vals[0]) * 0.25
        return None


class VisualizationPipeline:
    def __init__(self, config: AnalysisConfig):
        """Create a plotting helper that stores all figures in the configured output directory."""
        self.config = config

    def plot_auto_roi_debug(self, result: MtfResult):
        """Generate a side‑by‑side visualization of the auto‑ROI selection."""
        if not result.auto_roi_debug:
            return

        dbg = result.auto_roi_debug
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left Plot: Gradient Map
        axs[0].imshow(dbg.grad_image, cmap='viridis')
        axs[0].plot(dbg.edge_x_coords, dbg.edge_y_coords, 'r.', markersize=1, alpha=0.3, label='Top 0.5% Edges')
        axs[0].plot(dbg.center_x, dbg.center_y, 'cx', markersize=12, markeredgewidth=2, label='Median Center')
        axs[0].set_title("Gradient Map & Detected Edge Center")
        axs[0].legend()
        
        # Right Plot: Original Image + Cropping Box
        axs[1].imshow(dbg.full_image, cmap='gray')
        x1, x2, y1, y2 = dbg.roi_coords
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        axs[1].add_patch(rect)
        axs[1].plot(dbg.center_x, dbg.center_y, 'rx', markersize=10, markeredgewidth=2)
        axs[1].set_title(f"Final Auto-ROI Crop: {x2-x1}x{y2-y1}")
        
        out_path = self.config.output_dir / f"{result.name}_debug_auto_roi.png"
        plt.savefig(out_path)
        plt.close(fig)

    def plot_edge_debug(self, result: MtfResult):
        """Overlay the detected edge line on top of the ROI image and save the figure."""
        img = result.edge_debug_info.roi_image
        angle = result.edge_debug_info.edge_angle
        dist = result.edge_debug_info.edge_dist

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        h, w = img.shape
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        points = []

        # Intersections with borders
        if abs(sin_a) > 1e-8:
            y_left = (dist - 0 * cos_a) / sin_a
            y_right = (dist - w * cos_a) / sin_a
            if 0 <= y_left <= h:
                points.append((0, y_left))
            if 0 <= y_right <= h:
                points.append((w, y_right))

        if abs(cos_a) > 1e-8:
            x_top = (dist - 0 * sin_a) / cos_a
            x_bottom = (dist - h * sin_a) / cos_a
            if 0 <= x_top <= w:
                points.append((x_top, 0))
            if 0 <= x_bottom <= w:
                points.append((x_bottom, h))

        if len(points) >= 2:
            (x0, y0), (x1, y1) = points[:2]
            ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.set_title(f"Detected Edge: {np.rad2deg(angle):.1f}°")

        out_path = self.config.output_dir / f"{result.name}_debug_edge_overlay.png"
        plt.savefig(out_path)
        plt.close()

    def plot_single_analysis(self, result: MtfResult):
        """Plot ESF/LSF for a single result, including an optional FWHM marker."""
        esf_smooth = gaussian_filter1d(result.esf_raw, sigma=2.0)
        
        # Avoid division by zero in normalization
        esf_range = esf_smooth.max() - esf_smooth.min()
        if esf_range == 0: esf_range = 1.0
        esf_disp = (esf_smooth - esf_smooth.min()) / esf_range
        
        lsf_smooth = gaussian_filter1d(result.lsf_raw, sigma=2.0)
        lsf_max = max(lsf_smooth.max(), result.lsf_raw.max())
        if lsf_max == 0: lsf_max = 1.0
        lsf_disp = lsf_smooth / lsf_max

        fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        center = np.argmax(lsf_disp)
        window = 100
        start = max(0, center - window)
        end = min(len(esf_disp), center + window)
        bin_width = 0.25 # 4x oversampling
        x_axis = np.arange(len(esf_disp)) * bin_width

        axs[0].plot(x_axis[start:end], esf_disp[start:end], 'purple', lw=2)
        axs[0].plot(x_axis[start:end], result.esf_raw[start:end], 'k.', ms=1, alpha=0.3)
        axs[0].set_title(f'ESF - {result.name}')
        axs[0].grid(True, alpha=0.5)

        axs[1].plot(x_axis[start:end], lsf_disp[start:end], 'b-', lw=2)
        if result.fwhm_px:
            axs[1].axhline(0.5, color='r', linestyle='--', alpha=0.5, label=f"FWHM: {result.fwhm_px:.2f} px")
            axs[1].legend()
        
        axs[1].set_title('LSF')
        axs[1].grid(True, alpha=0.5)
        
        out_path = self.config.output_dir / f"{result.name}_analysis.pdf"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def plot_mtf_summary(self, results: List[MtfResult]):
        """Plot all MTF curves in a single comparison figure."""
        unit = "lp/mm" if self.config.use_lpmm else "cycles/pixel"

        fig, ax = plt.subplots(figsize=(10, 7))

        # Base Y at top of axes
        base_y = 1.01
        y_step = -0.03  # vertical spacing between labels
        current_stack_index = 0
        max_plot_limit = 0.5 # Default max

        for res in results:
            if res.mtf_raw.size < 2: 
                continue

            if self.config.use_lpmm and res.pixel_size_used:
                img_nyquist = 1 / (2 * (res.pixel_size_used / 1000.0))
                max_plot_limit = max(max_plot_limit, img_nyquist)

            # Cubic interpolation
            f_interp = interp1d(res.freqs_raw, res.mtf_raw, kind='cubic', bounds_error=False, fill_value=0)
            x_new = np.linspace(0, res.freqs_raw.max(), 500)
            y_new = np.clip(f_interp(x_new), 0, 110)

            # Smooth from 10% onward
            idx_start = np.argmax(y_new <= 90)
            y_smooth = y_new.copy()
            if idx_start < len(y_new) - 1:
                y_smooth[idx_start:] = savgol_filter(y_new[idx_start:], window_length=11, polyorder=3)

            # --- MTF50 ---
            if np.any(y_smooth >= 50) and self.config.mtf50:
                # Interpolate exact MTF50 frequency
                mtf50_freq = np.interp(50, y_smooth[::-1], x_new[::-1])

                # Transparent vertical line, not in legend
                ax.axvline(mtf50_freq, color='green', linestyle='--', alpha=0.5, label='_nolegend_')

                # Add stacked text
                ax.text(
                    mtf50_freq, base_y - current_stack_index * y_step,
                    f"MTF50: {mtf50_freq:.2f} {unit}",
                    color='green',
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    alpha=0.7,
                    transform=ax.get_xaxis_transform()
                )
                current_stack_index += 1

            # Curve label
            label = res.name
            if self.config.use_lpmm and res.pixel_size_used:
                label += f" ({res.pixel_size_used}µm)"

            ax.plot(x_new, y_smooth, label=label, lw=2)


        ax.set_xlabel(f"Spatial Frequency ({unit})")
        ax.set_ylabel("MTF (%)")
        ax.set_title("MTF Comparison")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max_plot_limit)

        ax.grid(True, which='both', linestyle='--')
        ax.legend()

        out_path = self.config.output_dir / "mtf_summary.pdf"
        plt.savefig(out_path)
        plt.show()

def main():
    """Entry point for the CLI interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default=".")
    parser.add_argument('--output', '-o', default='output')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug-plots', action='store_true')
    parser.add_argument('--lpmm', action='store_true')
    parser.add_argument('--MTF50', action='store_true')
    parser.add_argument('--default-pixel', type=float, default=3.76)
    parser.add_argument('--manual-roi', action='store_true')
    parser.add_argument('--esf-lsf-fwhm', action='store_true')
    args = parser.parse_args()

    config = AnalysisConfig(
        input_dir=Path(args.dir),
        output_dir=Path(args.output),
        verbose=args.verbose,
        use_lpmm=args.lpmm,
        default_pixel_size=args.default_pixel,
        show_debug_plots=args.debug_plots,
        mtf50=args.MTF50,
        manual_roi=args.manual_roi,
        esf_lsf_fwhm=args.esf_lsf_fwhm
    )
    config.setup_logging()
    config.output_dir.mkdir(exist_ok=True)

    processor = MtfProcessingPipeline(config)
    visualizer = VisualizationPipeline(config)

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp', '*.ARW', '*.CR2', '*.NEF', '*.DNG']
    images = []
    for ext in extensions:
        images.extend(config.input_dir.glob(ext))
        images.extend(config.input_dir.glob(ext.upper()))
    images = sorted(list(set(images)))

    if not images:
        print(f"No images found in {config.input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images.")
    results = []

    for img_path in images:
        print(f"\n--- Processing: {img_path.name} ---")
        current_pixel_size = None
        if config.use_lpmm:
            while True:
                user_input = input(f"Enter pixel size in µm for '{img_path.name}' [default: {config.default_pixel_size}]: ").strip()
                if user_input == "":
                    current_pixel_size = config.default_pixel_size
                    break
                try:
                    current_pixel_size = float(user_input)
                    break
                except ValueError:
                    print("Invalid number.")

        result = processor.run(img_path, pixel_size_um=current_pixel_size)
        if result:
            if config.show_debug_plots:
                visualizer.plot_auto_roi_debug(result)
                visualizer.plot_edge_debug(result)

            if config.esf_lsf_fwhm:
                visualizer.plot_single_analysis(result)
            results.append(result)
        else:
            print("Skipping due to error.")

    if results:
        #logging.debug(results)
        visualizer.plot_mtf_summary(results)
    else:
        print("No valid results.")

if __name__ == "__main__":
    main()