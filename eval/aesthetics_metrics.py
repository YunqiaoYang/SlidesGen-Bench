#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aesthetics Metrics Module for PowerPoint Slide Evaluation

This module calculates various aesthetic metrics for PowerPoint slides,
adapted from webpage aesthetics metrics originally designed for GUI evaluation.

Metrics included:
- Figure-ground contrast (m5)
- Subband entropy (m7)
- Feature congestion (m8) - visual clutter measure
- UMSI - Unified Model of Saliency and Importance (m9)
- LAB color space statistics (m14)
- HSV color space statistics (m16)
- Color harmony (m20)

Author: PPT Evaluation System
"""

import base64
import gc
import json
import logging
import math
import os
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy import signal
from skimage import transform as skit_transform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Score Parameters Configuration
# ----------------------------------------------------------------------------

@dataclass
class ScoreParameters:
    """Parameters for aesthetics score calculation.
    
    These parameters control how raw metrics are transformed into final scores.
    Can be loaded from a JSON configuration file.
    """
    # Figure-ground contrast
    weight_contrast: float = 20.0
    
    # Colorfulness
    weight_colorfulness: float = 1.0
    weight_colorfulness_score: float = 0.5
    w_colorfulness_mean: float = 50.0
    w_colorfulness_std: float = 20.0
    weight_colorfulness_pacing: float = 0.5
    w_colorfulness_target: float = 8.0
    w_colorfulness_target_std: float = 5.0
    
    # Color harmony
    weight_harmony: float = 0.5
    weight_harmony_mean: float = 100.0
    weight_harmony_deck: float = 100.0
    # Sigma for harmony normalization (S_slide = exp(-D^2 / (2*sigma^2)))
    # Range: 0.01 to 0.1, smaller sigma = stricter harmony requirement
    harmony_sigma: float = 0.05
    
    # Subband entropy
    weight_subband_entropy: float = 1.0
    weight_subband_entropy_score: float = 50.0
    w_subband_entropy_mean: float = 3.5
    w_subband_entropy_std: float = 0.5
    
    # Visual HRV
    weight_target_rmssd: float = 0.1
    weight_target_halfwidth: float = 0.1
    se_k: float = 1.5
    se_mu: float = 5.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScoreParameters':
        """Create parameters from dictionary."""
        # Filter only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)
    
    @classmethod
    def from_json_file(cls, config_path: str) -> 'ScoreParameters':
        """Load parameters from a JSON configuration file.
        
        The JSON file should have a "parameters" key containing the parameter values.
        Example:
        {
            "parameters": {
                "weight_contrast": 7,
                "weight_colorfulness": 9,
                ...
            }
        }
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            ScoreParameters instance with loaded values
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle nested "parameters" key
        if 'parameters' in data:
            params_data = data['parameters']
        else:
            params_data = data
        
        return cls.from_dict(params_data)
    
    def save_to_json_file(self, config_path: str) -> None:
        """Save parameters to a JSON configuration file.
        
        Args:
            config_path: Path to save the JSON configuration file
        """
        data = {"parameters": self.to_dict()}
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Global default parameters instance
_default_parameters = ScoreParameters()


def get_default_parameters() -> ScoreParameters:
    """Get the global default parameters instance."""
    return _default_parameters


def set_default_parameters(params: ScoreParameters) -> None:
    """Set the global default parameters instance."""
    global _default_parameters
    _default_parameters = params


def load_parameters_from_config(config_path: str) -> ScoreParameters:
    """Load parameters from config file and set as default.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Loaded ScoreParameters instance
    """
    params = ScoreParameters.from_json_file(config_path)
    set_default_parameters(params)
    logger.info(f"Loaded score parameters from {config_path}")
    return params


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to Base64 encoded string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_image_to_base64(img: Image.Image) -> str:
    """
    Convert a PIL Image to Base64 encoded PNG string.
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64 encoded string of the image
    """
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ----------------------------------------------------------------------------
# HSV Math Utility Functions (from m16/utils.py)
# ----------------------------------------------------------------------------

sind = lambda degrees: np.sin(np.deg2rad(degrees))
cosd = lambda degrees: np.cos(np.deg2rad(degrees))

atan2_func = lambda c, s: np.pi * (
    1.0
    - 0.5 * (1 + np.sign(c)) * (1 - np.sign(s**2))
    - 0.25 * (2 + np.sign(c)) * np.sign(s)
) - np.sign(c * s) * np.arctan(
    (np.abs(c) - np.abs(s)) / (np.abs(c) + np.abs(s))
)

atan2d = lambda c, s: atan2_func(c, s) * 180.0 / np.pi

np_sind = np.vectorize(sind)
np_cosd = np.vectorize(cosd)


# ----------------------------------------------------------------------------
# Color Harmony Utility Classes (from m20/utils.py)
# ----------------------------------------------------------------------------

def deg_distance(deg_arr: np.ndarray, deg_float: float) -> np.ndarray:
    """Compute absolute minimum distance between array elements and a float."""
    d1 = np.abs(deg_arr - deg_float)
    d2 = np.abs(360 - d1)
    return np.minimum(d1, d2)


class HueSector:
    """Represents a sector on the hue wheel."""
    
    def __init__(self, center: Union[int, float], width: Union[int, float]):
        self.center = center
        self.width = width
        self.border = [
            (self.center - self.width / 2),
            (self.center + self.width / 2),
        ]

    def is_in_sector(self, H: np.ndarray) -> np.ndarray:
        """Check if hue values are within this sector."""
        return deg_distance(H, self.center) < self.width / 2

    def distance_to_border(self, H: np.ndarray) -> np.ndarray:
        """Compute distance from hue values to the sector border."""
        d1 = deg_distance(H, self.border[0])
        d2 = deg_distance(H, self.border[1])
        return np.minimum(d1, d2)


class HarmonicScheme:
    """Represents a harmonic color scheme."""
    
    def __init__(self, sectors: List[HueSector]):
        self.sectors = sectors

    def harmony_score(self, X: np.ndarray) -> float:
        """
        Calculate the harmony score for an HSV image.
        
        Args:
            X: HSV image array
            
        Returns:
            Harmony score (lower is better)
        """
        H = X[:, :, 0].astype(np.float64) * 2  # OpenCV HSV: H is 0-179, convert to 0-358
        S = X[:, :, 1].astype(np.float64) / 255.0
        
        # For each pixel, find minimum distance to any sector
        min_dist = np.full(H.shape, 180.0)
        for sector in self.sectors:
            in_sector = sector.is_in_sector(H)
            dist = sector.distance_to_border(H)
            dist[in_sector] = 0
            min_dist = np.minimum(min_dist, dist)
        
        # Weight by saturation (more saturated colors matter more)
        weighted_dist = min_dist * S
        
        return float(np.mean(weighted_dist))

    def hue_shifted(self, X: np.ndarray, num_superpixels: int = -1) -> np.ndarray:
        """
        Shift hues to match the harmonic scheme.
        
        Args:
            X: HSV image array
            num_superpixels: Number of superpixels (-1 to disable)
            
        Returns:
            Shifted HSV image
        """
        result = X.copy()
        H = result[:, :, 0].astype(np.float64) * 2
        
        # Shift each pixel to nearest sector
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                hue = H[i, j]
                min_dist = 180
                target_hue = hue
                
                for sector in self.sectors:
                    if deg_distance(np.array([hue]), sector.center)[0] < sector.width / 2:
                        target_hue = hue
                        min_dist = 0
                        break
                    
                    # Check distance to borders
                    for border in sector.border:
                        dist = deg_distance(np.array([hue]), border)[0]
                        if dist < min_dist:
                            min_dist = dist
                            target_hue = border
                
                result[i, j, 0] = int(target_hue / 2) % 180
        
        return result


def count_hue_histogram(img_hsv: np.ndarray, bins: int = 360) -> np.ndarray:
    """Count hue histogram for an HSV image."""
    H = img_hsv[:, :, 0].astype(np.float64) * 2  # Convert to 0-358
    S = img_hsv[:, :, 1].astype(np.float64) / 255.0
    
    # Only count saturated pixels
    mask = S > 0.1
    H_masked = H[mask]
    
    hist, _ = np.histogram(H_masked, bins=bins, range=(0, 360))
    return hist


# ----------------------------------------------------------------------------
# Visual Clutter Utility Functions (for feature congestion m8)
# ----------------------------------------------------------------------------

def conv2(x: np.ndarray, y: np.ndarray, mode: Optional[str] = None) -> np.ndarray:
    """Computes the two-dimensional convolution of matrices x and y."""
    if mode == "same":
        return np.rot90(
            signal.convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2
        )
    else:
        return signal.convolve2d(x, y)


def addborder(im: np.ndarray, xbdr: int, ybdr: int, arg: Union[str, int, float]) -> np.ndarray:
    """Make image with added border."""
    if isinstance(arg, (int, float)):
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_CONSTANT, value=arg)
    elif arg == "even":
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_REFLECT)
    elif arg == "odd":
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_REFLECT_101)
    elif arg == "wrap":
        return cv2.copyMakeBorder(im, ybdr, ybdr, xbdr, xbdr, cv2.BORDER_WRAP)
    raise ValueError("unknown border style")


def filt2(kernel: np.ndarray, im1: np.ndarray, reflect_style: Union[str, int, float] = "odd") -> np.ndarray:
    """Improved version of filter2 with reflection."""
    ky, kx = kernel.shape
    iy, ix = im1.shape
    imbig = addborder(im1, kx, ky, reflect_style)
    imbig = conv2(imbig, kernel, "same")
    return imbig[ky:ky + iy, kx:kx + ix]


def RRoverlapconv(kernel: np.ndarray, in_: np.ndarray) -> np.ndarray:
    """Filters image with kernel, rescaling for overlap."""
    out = conv2(in_, kernel, mode="same")
    rect = np.ones_like(in_)
    overlapsum = conv2(rect, kernel, "same")
    return np.sum(kernel) * out / overlapsum


def RRgaussfilter1D(halfsupport: int, sigma: Union[int, float], center: Union[int, float] = 0) -> np.ndarray:
    """Creates a one-dimensional gaussian filter kernel."""
    t = list(range(-halfsupport, halfsupport + 1))
    kernel = np.array([np.exp(-((x - center) ** 2) / (2 * sigma**2)) for x in t])
    kernel = kernel / sum(kernel)
    return kernel.reshape(1, kernel.shape[0])


def DoG1filter(a: int, sigma: Union[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Creates 2 one-dimensional gaussian filters for DoG."""
    sigi = 0.71 * sigma
    sigo = 1.14 * sigma
    t = range(-a, a + 1)
    gi = np.array([np.exp(-(x**2) / (2 * sigi**2)) for x in t])
    gi = gi / np.sum(gi)
    go = np.array([np.exp(-(x**2) / (2 * sigo**2)) for x in t])
    go = go / np.sum(go)
    return gi.reshape(1, gi.shape[0]), go.reshape(1, go.shape[0])


def RRcontrast1channel(pyr: Dict, DoG_sigma: Union[int, float] = 2) -> List:
    """Filters a Gaussian pyramid with a 1-channel contrast feature detector."""
    levels = len(pyr)
    contrast = [0] * levels
    innerG1, outerG1 = DoG1filter(round(DoG_sigma * 3), DoG_sigma)
    
    for i in range(0, levels):
        inner = filt2(innerG1, pyr[(i, 0)])
        inner = filt2(innerG1.T, inner)
        outer = filt2(outerG1, pyr[(i, 0)])
        outer = filt2(outerG1.T, outer)
        contrast[i] = abs(inner - outer)
    return contrast


def reduce_image(image0: np.ndarray, kernel: Union[None, np.ndarray] = None) -> np.ndarray:
    """Reduce for building Gaussian or Laplacian pyramids."""
    if kernel is None:
        kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
    ysize, xsize = image0.shape
    image0 = filt2(kernel, image0)
    image1 = image0[:, range(0, xsize, 2)]
    image1 = filt2(kernel.T, image1)
    return image1[range(0, ysize, 2), :]


def RRoverlapconvexpand(in_: np.ndarray, kernel: np.ndarray = None) -> np.ndarray:
    """Expand image to double size."""
    if kernel is None:
        kernel = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
    ysize, xsize = in_.shape
    kernel = kernel * 2
    tmp = np.zeros([ysize, 2 * xsize])
    k = list(range(0, xsize))
    k_2 = [x * 2 for x in k]
    tmp[:, k_2] = in_[:, k]
    tmp = RRoverlapconv(kernel, tmp)
    out = np.zeros([2 * ysize, 2 * xsize])
    k = list(range(0, ysize))
    k_2 = [x * 2 for x in k]
    out[k_2, :] = tmp[k, :]
    return RRoverlapconv(kernel.T, out)


def HV(in_: list):
    """Outputs H-V, computes difference of first 2 elements."""
    return in_[0] - in_[1]


def DD(in_: list):
    """Outputs R-L, computes difference of last 2 elements."""
    return in_[3] - in_[2]


def sumorients(in_: list):
    """Sums the four orientations into one image."""
    return in_[0] + in_[1] + in_[2] + in_[3]


def poolnew(in_: list, sigma: Union[int, float, None] = None) -> Tuple:
    """Pools with a gaussian."""
    in1, in2, in3, in4 = in_[0], in_[1], in_[2], in_[3]
    if sigma is None:
        out1 = reduce_image(RRoverlapconvexpand(in1))
        out2 = reduce_image(RRoverlapconvexpand(in2))
        out3 = reduce_image(RRoverlapconvexpand(in3))
        out4 = reduce_image(RRoverlapconvexpand(in4))
    else:
        kernel = RRgaussfilter1D(round(2 * sigma), sigma)
        out1 = reduce_image(RRoverlapconvexpand(in1, kernel), kernel)
        out2 = reduce_image(RRoverlapconvexpand(in2, kernel), kernel)
        out3 = reduce_image(RRoverlapconvexpand(in3, kernel), kernel)
        out4 = reduce_image(RRoverlapconvexpand(in4, kernel), kernel)
    return out1, out2, out3, out4


def imrotate(im: np.ndarray, angle: Union[float, int], method: str = "bicubic", bbox: str = "crop") -> np.ndarray:
    """Rotate an image by angle (in degrees)."""
    func_method = {"nearest": 0, "bilinear": 1, "biquadratic": 2, "bicubic": 3, "biquartic": 4, "biquintic": 5}
    func_bbox = {"loose": True, "crop": False}
    immin = np.min(im)
    imrange = np.max(im) - immin
    im = (im - immin) / imrange if imrange > 0 else im - immin
    im = skit_transform.rotate(im, angle, order=func_method[method], resize=func_bbox[bbox])
    im = im * imrange + immin
    return im


def orient_filtnew(pyr: np.ndarray, sigma: Union[int, float] = 16 / 14) -> Tuple:
    """Filters pyr with 2nd derivative filters in 4 directions."""
    halfsupport = round(3 * sigma)
    sigy = sigma
    sigx = sigma
    
    gx = RRgaussfilter1D(halfsupport, sigx)
    gy = RRgaussfilter1D(halfsupport, sigy, sigma)
    Ga = conv2(gx, gy.T)
    Ga = Ga / np.sum(Ga)
    gy = RRgaussfilter1D(halfsupport, sigy)
    Gb = conv2(gx, gy.T)
    Gb = Gb / np.sum(Gb)
    gy = RRgaussfilter1D(halfsupport, sigy, -sigma)
    Gc = conv2(gx, gy.T)
    Gc = Gc / np.sum(Gc)
    H = -Ga + 2 * Gb - Gc
    V = H.T
    
    GGa = imrotate(Ga, 45, "bicubic", "crop")
    GGa = GGa / np.sum(GGa) if np.sum(GGa) != 0 else GGa
    GGb = imrotate(Gb, 45, "bicubic", "crop")
    GGb = GGb / np.sum(GGb) if np.sum(GGb) != 0 else GGb
    GGc = imrotate(Gc, 45, "bicubic", "crop")
    GGc = GGc / np.sum(GGc) if np.sum(GGc) != 0 else GGc
    R = -GGa + 2 * GGb - GGc
    GGa = imrotate(Ga, -45, "bicubic", "crop")
    GGa = GGa / np.sum(GGa) if np.sum(GGa) != 0 else GGa
    GGb = imrotate(Gb, -45, "bicubic", "crop")
    GGb = GGb / np.sum(GGb) if np.sum(GGb) != 0 else GGb
    GGc = imrotate(Gc, -45, "bicubic", "crop")
    GGc = GGc / np.sum(GGc) if np.sum(GGc) != 0 else GGc
    L = -GGa + 2 * GGb - GGc
    
    return filt2(H, pyr), filt2(V, pyr), filt2(L, pyr), filt2(R, pyr)


def normlize(arr: np.ndarray) -> np.ndarray:
    """Normalizes array between (min, max) -> (0, 255)."""
    min_min = arr.min()
    max_max = arr.max()
    if min_min == max_max:
        return np.full_like(arr, 255 / 2).astype("uint8")
    return ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")


def rgb2lab_clutter(im: np.ndarray) -> np.ndarray:
    """Converts RGB to CIELab color space (for clutter metrics)."""
    im = im / 255.0
    mask = im >= 0.04045
    im[mask] = ((im[mask] + 0.055) / 1.055) ** 2.4
    im[~mask] = im[~mask] / 12.92
    
    matrix = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ])
    c_im = np.dot(im, matrix.T)
    c_im[:, :, 0] = c_im[:, :, 0] / 95.047
    c_im[:, :, 1] = c_im[:, :, 1] / 100.000
    c_im[:, :, 2] = c_im[:, :, 2] / 108.833
    
    mask = c_im >= 0.008856
    c_im[mask] = c_im[mask] ** (1 / 3)
    c_im[~mask] = 7.787 * c_im[~mask] + 16 / 116
    
    im_Lab = np.zeros_like(c_im)
    im_Lab[:, :, 0] = (116 * c_im[:, :, 1]) - 16
    im_Lab[:, :, 1] = 500 * (c_im[:, :, 0] - c_im[:, :, 1])
    im_Lab[:, :, 2] = 200 * (c_im[:, :, 1] - c_im[:, :, 2])
    return im_Lab


# ----------------------------------------------------------------------------
# LAB Color Space Utilities (for m7 and m14)
# ----------------------------------------------------------------------------

def rgb2lab(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to LAB color space.
    
    Args:
        img_rgb: RGB image array (0-255)
        
    Returns:
        LAB image array
    """
    # Use OpenCV for conversion
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return img_lab


def entropy(data: np.ndarray, bins: int = 256) -> float:
    """
    Calculate Shannon entropy of data.
    
    Args:
        data: 1D array of values
        bins: Number of histogram bins (default 256)
        
    Returns:
        Shannon entropy value
    """
    # Flatten data
    data_flat = data.ravel()
    
    # Calculate histogram with proper normalization
    hist, _ = np.histogram(data_flat, bins=bins, density=False)
    
    # Normalize to get probabilities
    hist = hist.astype(np.float64)
    hist = hist / np.sum(hist)
    
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    # Calculate Shannon entropy
    return float(-np.sum(hist * np.log2(hist)))


# ----------------------------------------------------------------------------
# Aesthetic Metric Implementations
# ----------------------------------------------------------------------------

class FigureGroundContrastMetric:
    """
    Metric: Figure-ground contrast.
    
    Measures the difference in color or luminance between adjacent areas using
    WCAG relative luminance and contrast ratio standards.
    Higher values indicate better contrast between foreground and background.
    
    This implementation uses:
    - W3C relative luminance calculation
    - WCAG contrast ratio (1.0 to 21.0)
    - Logarithmic normalization to [0, 1] scale
    
    Returns a normalized score where:
    - 1.0 = maximum contrast (21:1, e.g., white on black)
    - ~0.49 = WCAG AA standard (~4.5:1)
    - ~0.35 = low contrast (poor readability)
    - 0.0 = no contrast (same color)
    """
    
    @staticmethod
    def _calculate_relative_luminance(rgb: np.ndarray) -> np.ndarray:
        """
        Calculate relative luminance using W3C standard formula.
        
        Args:
            rgb: RGB array with values in range [0, 255], shape (..., 3)
            
        Returns:
            Relative luminance array with values in range [0, 1]
        """
        # Normalize to [0, 1]
        rgb_normalized = rgb.astype(np.float64) / 255.0
        
        # Apply sRGB gamma correction
        linear_rgb = np.where(
            rgb_normalized <= 0.03928,
            rgb_normalized / 12.92,
            np.power((rgb_normalized + 0.055) / 1.055, 2.4)
        )
        
        # Calculate luminance using ITU-R BT.709 coefficients
        luminance = (
            0.2126 * linear_rgb[..., 0] +
            0.7152 * linear_rgb[..., 1] +
            0.0722 * linear_rgb[..., 2]
        )
        
        return luminance
    
    @staticmethod
    def _calculate_contrast_ratio(lum1: float, lum2: float) -> float:
        """
        Calculate WCAG contrast ratio between two luminance values.
        
        Args:
            lum1: First luminance value [0, 1]
            lum2: Second luminance value [0, 1]
            
        Returns:
            Contrast ratio in range [1.0, 21.0]
        """
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        return (lighter + 0.05) / (darker + 0.05)
    
    @staticmethod
    def _normalize_contrast_score(contrast_ratio: float) -> float:
        """
        Normalize contrast ratio to [0, 1] using logarithmic mapping.
        
        Logarithmic scale is used because human perception of contrast
        is logarithmic, not linear. This ensures that meaningful contrasts
        like 4.5:1 (WCAG AA) are properly valued.
        
        Args:
            contrast_ratio: Contrast ratio [1.0, 21.0]
            
        Returns:
            Normalized score [0.0, 1.0]
        """
        import math
        max_contrast = 21.0
        score = math.log(contrast_ratio) / math.log(max_contrast)
        return max(0.0, min(score, 1.0))
    
    @staticmethod
    def _load_detection_boxes(image_path: str) -> Optional[List[Dict]]:
        """
        Load detection boxes from JSON file based on image path.
        
        Converts image path like:
        .../slide_images/slide_0001.png
        to detection JSON path:
        .../detection/slide_0001.json
        
        Args:
            image_path: Path to the slide image
            
        Returns:
            List of detection boxes or None if file not found
        """
        import json
        from pathlib import Path
        
        try:
            img_path = Path(image_path)
            
            # Replace 'slide_images' with 'detection' and change extension to .json
            detection_dir = img_path.parent.parent / 'detection'
            detection_file = detection_dir / (img_path.stem + '.json')
            
            if not detection_file.exists():
                logger.debug(f"Detection file not found: {detection_file}")
                return None
            
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            boxes = data.get('boxes', [])
            if boxes:
                logger.info(f"Loaded {len(boxes)} detection boxes from {detection_file}")
                return boxes
            else:
                logger.debug(f"No boxes found in {detection_file}")
                return None
                
        except Exception as e:
            logger.debug(f"Failed to load detection boxes from {image_path}: {e}")
            return None

    @classmethod
    def execute(cls, img: Image.Image, detection_boxes: Optional[List[Dict]] = None, 
                gui_type: int = 0, image_path: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Execute the figure-ground contrast metric.
        
        If detection_boxes are provided or can be loaded from image_path, analyzes text-background contrast:
        1. Extract text regions from bounding boxes
        2. Compare text color with background in each box
        3. Return min, max, and mean contrast scores
        
        If no detection_boxes provided/found, uses grid-based analysis (legacy).
        
        Args:
            img: PIL Image
            detection_boxes: List of detection boxes with format:
                [{"label": str, "coordinate": [x1, y1, x2, y2], ...}, ...]
            gui_type: Unused (kept for backward compatibility)
            image_path: Path to the image file (used to auto-load detection boxes)
            
        Returns:
            If detection_boxes provided/found: Dict with "min", "max", "mean" scores
            Otherwise: Single float score (backward compatible)
        """
        # Convert to RGB array
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb)
        
        # Try to auto-load detection boxes if not provided
        if detection_boxes is None and image_path is not None:
            detection_boxes = cls._load_detection_boxes(image_path)
        
        # If detection boxes are available, use text-based contrast analysis
        if detection_boxes is not None:
            return cls._execute_with_detection(img_array, detection_boxes)
        
        # Legacy grid-based analysis
        return cls._execute_grid_based(img_array)
    
    @classmethod
    def _execute_with_detection(cls, img_array: np.ndarray, 
                                 detection_boxes: List[Dict]) -> Dict[str, float]:
        """
        Calculate text-background contrast using detection boxes.
        
        Args:
            img_array: RGB image array
            detection_boxes: List of detection results
            
        Returns:
            Dictionary with min, max, mean contrast scores
        """
        # Filter for text boxes only
        text_boxes = [box for box in detection_boxes if box.get("label") in ["text","doc_title","figure_title","footer","paragraph_title"]]
        
        if not text_boxes:
            logger.warning("No text boxes found in detection results")
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        
        contrast_scores = []
        h, w = img_array.shape[:2]
        
        for box in text_boxes:
            coords = box.get("coordinate", [])
            if len(coords) != 4:
                continue
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = coords
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract the text box region
            text_region = img_array[y1:y2, x1:x2]
            
            if text_region.size == 0:
                continue
            
            # Calculate luminance for the text region
            text_luminance = cls._calculate_relative_luminance(text_region)
            
            # Strategy: Use percentile-based approach to separate text from background
            # Text boxes typically have bimodal distribution: dark text on light bg OR light text on dark bg
            # We sample the darkest and lightest pixels and compute their contrast
            
            # Flatten luminance array for percentile calculation
            lum_flat = text_luminance.flatten()
            
            # Use 10th and 90th percentiles to avoid outliers
            # The larger group (more pixels) is likely the background
            # The smaller group (less pixels) is likely the text
            dark_threshold = np.percentile(lum_flat, 10)
            light_threshold = np.percentile(lum_flat, 90)
            
            # Calculate average luminance for dark and light regions
            dark_pixels = lum_flat[lum_flat <= dark_threshold]
            light_pixels = lum_flat[lum_flat >= light_threshold]
            
            if len(dark_pixels) > 0 and len(light_pixels) > 0:
                dark_lum_avg = np.mean(dark_pixels)
                light_lum_avg = np.mean(light_pixels)
                
                # Calculate contrast ratio between dark and light regions
                ratio = cls._calculate_contrast_ratio(dark_lum_avg, light_lum_avg)
                score = cls._normalize_contrast_score(ratio)
                contrast_scores.append(score)
                
                # Debug logging for troubleshooting
                logger.debug(f"Box {len(contrast_scores)}: dark_lum={dark_lum_avg:.3f}, "
                           f"light_lum={light_lum_avg:.3f}, ratio={ratio:.2f}, score={score:.3f}")
            else:
                # Fallback: insufficient variation in the region
                logger.debug(f"Box skipped: insufficient luminance variation")
        
        if not contrast_scores:
            logger.warning("No valid text boxes processed")
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        
        return {
            "min": float(np.min(contrast_scores)),
            "max": float(np.max(contrast_scores)),
            "mean": float(np.mean(contrast_scores)),
        }
    
    @classmethod
    def _execute_grid_based(cls, img_array: np.ndarray) -> float:
        """
        Legacy grid-based contrast analysis.
        
        Args:
            img_array: RGB image array
            
        Returns:
            Average contrast score
        """
        # Calculate luminance for entire image
        luminance = cls._calculate_relative_luminance(img_array)
        
        # Simple segmentation: divide image into grid and compare adjacent regions
        # This captures local contrast across the slide
        grid_size = 4  # 4x4 grid
        h, w = luminance.shape
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        contrast_scores = []
        
        # Compare adjacent cells horizontally and vertically
        for i in range(grid_size):
            for j in range(grid_size):
                # Get current cell
                y1, y2 = i * cell_h, min((i + 1) * cell_h, h)
                x1, x2 = j * cell_w, min((j + 1) * cell_w, w)
                
                cell_lum = luminance[y1:y2, x1:x2]
                cell_avg = np.mean(cell_lum)
                
                # Compare with right neighbor
                if j < grid_size - 1:
                    x1_r, x2_r = (j + 1) * cell_w, min((j + 2) * cell_w, w)
                    right_lum = luminance[y1:y2, x1_r:x2_r]
                    right_avg = np.mean(right_lum)
                    
                    ratio = cls._calculate_contrast_ratio(cell_avg, right_avg)
                    score = cls._normalize_contrast_score(ratio)
                    contrast_scores.append(score)
                
                # Compare with bottom neighbor
                if i < grid_size - 1:
                    y1_b, y2_b = (i + 1) * cell_h, min((i + 2) * cell_h, h)
                    bottom_lum = luminance[y1_b:y2_b, x1:x2]
                    bottom_avg = np.mean(bottom_lum)
                    
                    ratio = cls._calculate_contrast_ratio(cell_avg, bottom_avg)
                    score = cls._normalize_contrast_score(ratio)
                    contrast_scores.append(score)
        
        # Return average contrast score
        if contrast_scores:
            return float(np.mean(contrast_scores))
        else:
            return 0.0
    
    @classmethod
    def compute_score(
        cls,
        contrast_mean: float,
        params: Optional[ScoreParameters] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute figure-ground contrast score component.
        
        Based on main.py formula:
        contrast_score = weight_contrast * contrast_mean
        
        Args:
            contrast_mean: Mean contrast value across slides
            params: Score parameters (uses default if None)
            
        Returns:
            Tuple of (total_score, component_dict)
        """
        if params is None:
            params = get_default_parameters()
        
        contrast_score = params.weight_contrast * contrast_mean
        
        return contrast_score, {
            "contrast_total": contrast_score,
            "contrast_mean_raw": contrast_mean,
        }


class LABMetric:
    """
    Metric: LAB color space statistics.
    
    Computes average and standard deviation for L, A, B channels.
    """

    @classmethod
    def execute(cls, img: Image.Image) -> Dict[str, float]:
        """
        Execute the LAB metric.
        
        Args:
            img: PIL Image
            
        Returns:
            Dictionary with L_avg, L_std, A_avg, A_std, B_avg, B_std
        """
        # Convert to RGB
        img_rgb = img.convert("RGB")
        img_rgb_nparray = np.array(img_rgb)
        
        # Convert to LAB using skimage for better accuracy
        try:
            from skimage import color
            lab = color.rgb2lab(img_rgb_nparray, illuminant="D65", observer="2")
        except ImportError:
            # Fallback to OpenCV
            img_bgr = cv2.cvtColor(img_rgb_nparray, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            # OpenCV LAB ranges: L: 0-255, A: 0-255, B: 0-255
            # Convert to standard ranges: L: 0-100, A: -128 to 127, B: -128 to 127
            lab[:, :, 0] = lab[:, :, 0] * 100 / 255
            lab[:, :, 1] = lab[:, :, 1] - 128
            lab[:, :, 2] = lab[:, :, 2] - 128
        
        L = lab[:, :, 0]
        A = lab[:, :, 1]
        B = lab[:, :, 2]
        
        return {
            "L_avg": float(np.mean(L)),
            "L_std": float(np.std(L)),
            "A_avg": float(np.mean(A)),
            "A_std": float(np.std(A)),
            "B_avg": float(np.mean(B)),
            "B_std": float(np.std(B)),
        }


class HSVMetric:
    """
    Metric: HSV color space statistics.
    
    Computes average and standard deviation for H, S, V channels.
    """

    @classmethod
    def execute(cls, img: Image.Image) -> Dict[str, float]:
        """
        Execute the HSV metric.
        
        Args:
            img: PIL Image
            
        Returns:
            Dictionary with H_avg, S_avg, S_std, V_avg, V_std
        """
        # Convert to HSV
        img_hsv = img.convert("HSV")
        img_hsv_nparray = np.array(img_hsv) / 255.0
        
        img_hue = img_hsv_nparray[:, :, 0] * 359.0
        img_saturation = img_hsv_nparray[:, :, 1]
        img_value = img_hsv_nparray[:, :, 2]
        
        # Hue is circular, compute circular mean using native numpy (much faster)
        hue_rad = np.deg2rad(img_hue)
        hue_avg_sin = float(np.mean(np.sin(hue_rad)))
        hue_avg_cos = float(np.mean(np.cos(hue_rad)))
        hue_avg = float(atan2d(hue_avg_cos, hue_avg_sin))
        
        return {
            "H_avg": hue_avg,
            "S_avg": float(np.mean(img_saturation)),
            "S_std": float(np.std(img_saturation)),
            "V_avg": float(np.mean(img_value)),
            "V_std": float(np.std(img_value)),
        }


def _compute_harmony_for_template_alpha(args: Tuple) -> Tuple[int, int, float]:
    """
    Helper function for parallel harmony computation.
    
    Args:
        args: Tuple of (template_idx, alpha, template_sectors_data, img_hsv_arr)
        
    Returns:
        Tuple of (template_idx, alpha, harmony_score)
    """
    template_idx, alpha, template_data, img_hsv_arr = args
    sectors = []
    for t in template_data:
        center = t[0] * 360 + alpha
        width = t[1] * 360
        sectors.append(HueSector(center, width))
    scheme = HarmonicScheme(sectors)
    score = scheme.harmony_score(img_hsv_arr)
    return (template_idx, alpha, score)


def _compute_harmony_vectorized(H: np.ndarray, S: np.ndarray, 
                                 template_data: List[Tuple[float, float]], 
                                 N: int = 360) -> np.ndarray:
    """
    Memory-efficient vectorized computation of harmony scores.
    
    Uses chunked processing to avoid creating huge (N, n_pixels) arrays.
    Formula: Harmony Score = sum(S_p * dist(H_p, T_alpha)) / sum(S_p)
    
    Args:
        H: Hue array (0-358 range)
        S: Saturation array (0-1 range)
        template_data: List of (center_ratio, width_ratio) tuples
        N: Number of angles to evaluate
        
    Returns:
        Array of harmony scores for each angle
    """
    # Flatten H and S
    H_flat = H.ravel()
    S_flat = S.ravel()
    n_pixels = len(H_flat)
    
    # Compute sum of saturation for normalization
    sum_saturation = np.sum(S_flat)
    if sum_saturation == 0:
        # If no saturated pixels, return zeros
        return np.zeros(N, dtype=np.float64)
    
    # Create angle array
    alphas = np.arange(N, dtype=np.float64)  # (N,)
    
    # Process in chunks to limit memory usage
    # Target: ~100MB per chunk (N * chunk_size * 8 bytes * 4 arrays for safety)
    max_chunk_memory = 100_000_000  # 100MB
    chunk_size = min(n_pixels, max(5000, max_chunk_memory // (N * 8 * 4)))
    n_chunks = (n_pixels + chunk_size - 1) // chunk_size
    
    # Accumulate weighted sums
    weighted_sum = np.zeros(N, dtype=np.float64)
    
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_pixels)
        
        H_chunk = H_flat[start:end]
        S_chunk = S_flat[start:end]
        chunk_len = end - start
        
        # Initialize minimum distances for this chunk
        min_dist_chunk = np.full((N, chunk_len), 180.0, dtype=np.float64)
        
        for t in template_data:
            center_ratio, width_ratio = t
            width = width_ratio * 360
            half_width = width / 2
            
            # Centers for all angles
            centers = center_ratio * 360 + alphas  # (N,)
            
            # Distance to center: (N, chunk_len)
            # Use proper circular distance: min(|d|, |360 - |d||)
            diff = centers[:, None] - H_chunk[None, :]
            d1 = np.abs(diff)
            dist_to_center = np.minimum(d1, np.abs(360 - d1))
            
            # Check if in sector
            in_sector = dist_to_center < half_width
            
            # Distance to borders
            border1 = centers - half_width
            border2 = centers + half_width
            
            diff1 = border1[:, None] - H_chunk[None, :]
            d1_b1 = np.abs(diff1)
            dist_to_border1 = np.minimum(d1_b1, np.abs(360 - d1_b1))
            
            diff2 = border2[:, None] - H_chunk[None, :]
            d1_b2 = np.abs(diff2)
            dist_to_border2 = np.minimum(d1_b2, np.abs(360 - d1_b2))
            
            dist_to_border = np.minimum(dist_to_border1, dist_to_border2)
            dist_to_border[in_sector] = 0
            
            min_dist_chunk = np.minimum(min_dist_chunk, dist_to_border)
        
        # Accumulate weighted distances
        weighted_sum += np.sum(min_dist_chunk * S_chunk[None, :], axis=1)
    
    # Normalize by sum of saturation (not number of pixels)
    # Formula: sum(S_p * dist(H_p, T_alpha)) / sum(S_p)
    scores = weighted_sum / sum_saturation
    
    return scores


def _compute_harmony_for_template_vectorized(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Compute harmony scores for a single template across all angles using vectorization.
    
    Args:
        args: Tuple of (template_idx, template_data, H, S, N)
        
    Returns:
        Tuple of (template_idx, scores_array)
    """
    template_idx, template_data, H, S, N = args
    scores = _compute_harmony_vectorized(H, S, template_data, N)
    return (template_idx, scores)


def _compute_harmony_for_template(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Compute harmony scores for a single template across all angles.
    
    Args:
        args: Tuple of (template_idx, template_data, img_hsv_arr, N)
        
    Returns:
        Tuple of (template_idx, scores_array)
    """
    template_idx, template_data, img_hsv_arr, N = args
    scores = np.zeros(N)
    for alpha in range(N):
        sectors = []
        for t in template_data:
            center = t[0] * 360 + alpha
            width = t[1] * 360
            sectors.append(HueSector(center, width))
        scheme = HarmonicScheme(sectors)
        scores[alpha] = scheme.harmony_score(img_hsv_arr)
    return (template_idx, scores)


def _compute_harmony_chunk(args: Tuple) -> Tuple[int, int, int, np.ndarray]:
    """
    Compute harmony scores for a chunk of angles for a single template.
    
    Args:
        args: Tuple of (template_idx, start_alpha, end_alpha, template_data, H, S)
        
    Returns:
        Tuple of (template_idx, start_alpha, end_alpha, scores_array)
    """
    template_idx, start_alpha, end_alpha, template_data, H, S = args
    chunk_size = end_alpha - start_alpha
    scores = _compute_harmony_vectorized(H, S, template_data, N=360)
    return (template_idx, start_alpha, end_alpha, scores[start_alpha:end_alpha])


class ColorHarmonyMetric:
    """
    Metric: Color harmony.
    
    Computes distance to the closest harmonic color template.
    Lower values indicate better color harmony.
    """
    
    # Harmonic templates based on color theory
    HUE_TEMPLATES = {
        "i": [(0.00, 0.05)],
        "V": [(0.00, 0.26)],
        "L": [(0.00, 0.05), (0.25, 0.22)],
        "mirror_L": [(0.00, 0.05), (-0.25, 0.22)],
        "I": [(0.00, 0.05), (0.50, 0.05)],
        "T": [(0.25, 0.50)],
        "Y": [(0.00, 0.26), (0.50, 0.05)],
        "X": [(0.00, 0.26), (0.50, 0.26)],
    }

    @classmethod
    def _get_sectors(cls, template_type: str, alpha: float) -> List[HueSector]:
        """Create HueSectors for a template at a given angle."""
        sectors = []
        for t in cls.HUE_TEMPLATES[template_type]:
            center = t[0] * 360 + alpha
            width = t[1] * 360
            sectors.append(HueSector(center, width))
        return sectors

    @classmethod
    def execute(cls, img: Image.Image, n_workers: int = None, use_vectorized: bool = True, 
                max_pixels: int = 100000) -> Dict[str, Any]:
        """
        Execute the color harmony metric with parallel computation.
        
        Args:
            img: PIL Image
            n_workers: Number of parallel workers (default: number of CPU cores)
            use_vectorized: Use vectorized computation (much faster, default True)
            max_pixels: Maximum pixels to process (image will be resized if larger)
            
        Returns:
            Dictionary with best_template, best_distance, and distances for all templates
        """
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        # Resize image if too large to reduce computation
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        n_pixels = width * height
        
        if n_pixels > max_pixels:
            # Resize while preserving aspect ratio
            scale = (max_pixels / n_pixels) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = img_rgb.resize((new_width, new_height), Image.LANCZOS)
        
        img_rgb_arr = np.array(img_rgb)
        img_hsv_arr = cv2.cvtColor(img_rgb_arr, cv2.COLOR_RGB2HSV)
        
        # Pre-compute H and S arrays for vectorized computation
        H = img_hsv_arr[:, :, 0].astype(np.float64) * 2  # Convert to 0-358 range
        S = img_hsv_arr[:, :, 1].astype(np.float64) / 255.0
        
        N = 360
        template_types = list(cls.HUE_TEMPLATES.keys())
        F_matrix = np.zeros((len(template_types), N))
        
        # Determine number of workers
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), len(template_types))
        
        if use_vectorized:
            # Vectorized parallel computation - much faster
            # Prepare arguments for parallel execution
            template_args = [
                (i, cls.HUE_TEMPLATES[template], H, S, N)
                for i, template in enumerate(template_types)
            ]
            
            # Use ThreadPoolExecutor for better performance with numpy arrays
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_compute_harmony_for_template_vectorized, template_args))
            
            # Collect results
            for template_idx, scores in results:
                F_matrix[template_idx, :] = scores
        else:
            # Original non-vectorized parallel computation
            template_args = [
                (i, cls.HUE_TEMPLATES[template], img_hsv_arr, N)
                for i, template in enumerate(template_types)
            ]
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_compute_harmony_for_template, template_args))
            
            for template_idx, scores in results:
                F_matrix[template_idx, :] = scores
        
        # Find best template
        best_idx, best_alpha = np.unravel_index(np.argmin(F_matrix), F_matrix.shape)
        best_template = template_types[best_idx]
        best_distance = float(F_matrix[best_idx, best_alpha])
        
        # Compute best distance for each template
        template_distances = {}
        for i, template in enumerate(template_types):
            template_distances[template] = float(np.min(F_matrix[i]))
        
        return {
            "best_template": best_template,
            "best_distance": best_distance,
            "template_distances": template_distances,
        }
    
    @classmethod
    def compute_slide_score(cls, best_distance: float, sigma: float = 12.0) -> float:
        """
        Compute normalized slide score using Gaussian (Energy-Based) decay.
        
        Converts the saturation-weighted angular error into a 0-1 quality probability:
        S_slide = exp(- D_bar^2 / (2 * sigma^2))
        
        Args:
            best_distance: The weighted average hue distance (D_bar)
            sigma: Tolerance parameter (default: 12.0 for strict professional design)
            
        Returns:
            Normalized score between 0 and 1
        """
        return float(np.exp(- (best_distance ** 2) / (2 * sigma ** 2)))
    
    @classmethod
    def compute_deck_metrics(cls, slide_scores: List[float]) -> Dict[str, float]:
        """
        Compute deck-level consistency and final composite score.
        
        Uses deck consistency (standard deviation) defined as:
        σ_deck = sqrt(1/N * sum((S_slide_i - μ)^2))
        
        Args:
            slide_scores: List of normalized slide scores (0-1 range), in presentation order
            
        Returns:
            Dictionary containing:
            - mean_score: Average slide score (μ)
            - deck_consistency: Standard deviation σ_deck
            - final_score: Composite score (0-100) that rewards high average 
                          harmony and penalizes inconsistency
        """
        if not slide_scores:
            return {
                "mean_score": 0.0,
                "deck_consistency": 0.0,
                "final_score": 0.0,
            }
        
        if len(slide_scores) == 1:
            return {
                "mean_score": float(slide_scores[0]),
                "deck_consistency": 0.0,
                "final_score": 100 * float(slide_scores[0]),
            }
        
        slide_scores_arr = np.array(slide_scores)
        mean_score = float(np.mean(slide_scores_arr))
        
        # Deck consistency: σ_deck = sqrt(1/N * sum((S_i - μ)^2))
        # This is the standard deviation
        deck_consistency = float(np.std(slide_scores_arr))
        
        # Final score: 100 * mean - 50 * σ_deck
        # Rewards high average harmony and penalizes inconsistency
        final_score = 100 * mean_score - 100 * deck_consistency
        
        return {
            "mean_score": mean_score,
            "deck_consistency": deck_consistency,
            "final_score": final_score,
        }
    
    @classmethod
    def execute_sequential(cls, img: Image.Image) -> Dict[str, Any]:
        """
        Execute the color harmony metric sequentially (non-parallel version).
        
        Args:
            img: PIL Image
            
        Returns:
            Dictionary with best_template, best_distance, and distances for all templates
        """
        # Convert to HSV
        img_rgb = img.convert("RGB")
        img_rgb_arr = np.array(img_rgb)
        img_hsv_arr = cv2.cvtColor(img_rgb_arr, cv2.COLOR_RGB2HSV)
        
        N = 360
        template_types = list(cls.HUE_TEMPLATES.keys())
        F_matrix = np.zeros((len(template_types), N))
        
        # Compute harmony scores for all templates and angles
        for i, template in enumerate(template_types):
            for alpha in range(N):
                sectors = cls._get_sectors(template, alpha)
                scheme = HarmonicScheme(sectors)
                F_matrix[i, alpha] = scheme.harmony_score(img_hsv_arr)
        
        # Find best template
        best_idx, best_alpha = np.unravel_index(np.argmin(F_matrix), F_matrix.shape)
        best_template = template_types[best_idx]
        best_distance = float(F_matrix[best_idx, best_alpha])
        
        # Compute best distance for each template
        template_distances = {}
        for i, template in enumerate(template_types):
            template_distances[template] = float(np.min(F_matrix[i]))
        
        return {
            "best_template": best_template,
            "best_distance": best_distance,
            "template_distances": template_distances,
        }
    
    @classmethod
    def compute_slide_score_with_params(
        cls,
        best_distance: float,
        params: Optional[ScoreParameters] = None
    ) -> float:
        """
        Compute normalized slide score using configurable sigma parameter.
        
        Based on main.py formula:
        S_slide = exp(-D^2 / (2 * sigma^2))
        
        Args:
            best_distance: The weighted average hue distance (D_bar)
            params: Score parameters (uses default if None)
            
        Returns:
            Normalized score between 0 and 1
        """
        if params is None:
            params = get_default_parameters()
        
        sigma = params.harmony_sigma
        return float(math.exp(-(best_distance ** 2) / (2 * sigma ** 2)))
    
    @classmethod
    def compute_score(
        cls,
        raw_distances: List[float],
        params: Optional[ScoreParameters] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute color harmony score component from raw distances.
        
        Based on main.py formula:
        - Compute slide scores from raw best_distance using configurable sigma
        - harmony_score = weight * (weight_mean * mean_score - weight_deck * deck_consistency)
        
        Args:
            raw_distances: List of best_distance values from each slide
            params: Score parameters (uses default if None)
            
        Returns:
            Tuple of (total_score, component_dict)
        """
        if params is None:
            params = get_default_parameters()
        
        sigma = params.harmony_sigma
        
        # Compute normalized slide scores from raw distances
        slide_scores = []
        for best_distance in raw_distances:
            # S_slide = exp(-D^2 / (2*sigma^2))
            slide_score = math.exp(-(best_distance ** 2) / (2 * sigma ** 2))
            slide_scores.append(slide_score)
        
        # Handle empty case
        if not slide_scores:
            return 0.0, {
                "harmony_total": 0.0,
                "mean_score": 0.0,
                "deck_consistency": 0.0,
            }
        
        slide_scores_arr = np.array(slide_scores)
        mean_score = float(np.mean(slide_scores_arr))
        deck_consistency = float(np.std(slide_scores_arr)) if len(slide_scores) > 1 else 0.0
        
        # Compute harmony score: weight * (weight_mean * mean - weight_deck * std)
        harmony_score = params.weight_harmony * (
            params.weight_harmony_mean * mean_score -
            params.weight_harmony_deck * deck_consistency
        )
        
        return harmony_score, {
            "harmony_total": harmony_score,
            "harmony_mean_component": params.weight_harmony * params.weight_harmony_mean * mean_score,
            "harmony_consistency_penalty": params.weight_harmony * params.weight_harmony_deck * deck_consistency,
            "mean_score": mean_score,
            "deck_consistency": deck_consistency,
            "slide_scores": slide_scores,
        }


class VisualComplexityMetric:
    """
    Metric: Visual complexity based on edge detection.
    
    Measures the visual complexity of an image using edge density.
    """

    @classmethod
    def execute(cls, img: Image.Image) -> Dict[str, float]:
        """
        Execute the visual complexity metric.
        
        Args:
            img: PIL Image
            
        Returns:
            Dictionary with edge_density and complexity_score
        """
        # Convert to grayscale
        img_gray = img.convert("L")
        img_arr = np.array(img_gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_arr, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        total_pixels = edges.size
        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / total_pixels
        
        # Calculate Laplacian variance (another complexity measure)
        laplacian = cv2.Laplacian(img_arr, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        
        return {
            "edge_density": float(edge_density),
            "laplacian_variance": laplacian_var,
            "complexity_score": float(edge_density * 100 + laplacian_var / 100),
        }


class ColorfulnessMetric:
    """
    Metric: Colorfulness.
    
    Measures how colorful an image is based on Hasler & Süsstrunk's method.
    """

    @classmethod
    def execute(cls, img: Image.Image) -> float:
        """
        Execute the colorfulness metric.
        
        Args:
            img: PIL Image
            
        Returns:
            Colorfulness score
        """
        # Convert to RGB array
        img_rgb = img.convert("RGB")
        img_arr = np.array(img_rgb).astype(np.float64)
        
        R = img_arr[:, :, 0]
        G = img_arr[:, :, 1]
        B = img_arr[:, :, 2]
        
        # Compute rg and yb
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # Compute mean and standard deviation
        rg_mean = np.mean(rg)
        rg_std = np.std(rg)
        yb_mean = np.mean(yb)
        yb_std = np.std(yb)
        
        # Compute colorfulness
        std_root = np.sqrt(rg_std**2 + yb_std**2)
        mean_root = np.sqrt(rg_mean**2 + yb_mean**2)
        colorfulness = std_root + 0.3 * mean_root
        
        return float(colorfulness)
    
    @classmethod
    def compute_score(
        cls,
        color_mean: float,
        color_std: float,
        params: Optional[ScoreParameters] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute colorfulness score component using Gaussian normalization.
        
        Based on main.py formula:
        - Gaussian for mean colorfulness: exp(-((mean - target_mean)^2) / (2 * target_std^2))
        - Gaussian for pacing (std): exp(-((std - target)^2) / (2 * target_std^2))
        
        Args:
            color_mean: Mean colorfulness value across slides
            color_std: Standard deviation of colorfulness across slides
            params: Score parameters (uses default if None)
            
        Returns:
            Tuple of (total_score, component_dict)
        """
        if params is None:
            params = get_default_parameters()
        
        # Gaussian for mean colorfulness
        color_score_mean = math.exp(
            -((color_mean - params.w_colorfulness_mean) ** 2) / (2 * params.w_colorfulness_std ** 2)
        )
        
        # Gaussian for pacing (std)
        color_score_pacing = math.exp(
            -((color_std - params.w_colorfulness_target) ** 2) / (2 * params.w_colorfulness_target_std ** 2)
        )
        
        # Combined score
        colorfulness_score = params.weight_colorfulness * (
            params.weight_colorfulness_score * color_score_mean +
            params.weight_colorfulness_pacing * color_score_pacing
        )
        
        return colorfulness_score, {
            "colorfulness_total": colorfulness_score,
            "colorfulness_mean_component": params.weight_colorfulness * params.weight_colorfulness_score * color_score_mean,
            "colorfulness_pacing_component": params.weight_colorfulness * params.weight_colorfulness_pacing * color_score_pacing,
            "color_score_mean_normalized": color_score_mean,
            "color_score_pacing_normalized": color_score_pacing,
        }


class SubbandEntropyMetric:
    """
    Metric: Subband entropy.
    
    Measures the amount of visual clutter based on subband decomposition.
    Based on visual clutter research by Rosenholtz et al. (2007).
    Requires pyrtools library (macOS/Linux only).
    
    The metric decomposes the image into subbands using a steerable pyramid,
    computes Shannon entropy for each subband, and combines them with weights
    that emphasize luminance over chrominance.
    
    Formula (from Rosenholtz et al., 2007, pages 7-8):
    SE = 0.84 * Σ(H(S_L)) + 0.08 * Σ(H(S_a)) + 0.08 * Σ(H(S_b))
    
    where H(S_i) is the Shannon entropy of subband i, computed using
    adaptive binning (bins = sqrt(number of coefficients in subband)).
    
    Category: Visual complexity > Information amount > Visual clutter.
    
    Reference: Rosenholtz, R., Li, Y., & Nakano, L. (2007). 
    Measuring visual clutter. Journal of Vision, 7(2), 17.
    """
    
    # Constants
    _W_LEVELS = 3  # Number of spatial scales for subband decomposition
    _WGHT_LUM = 0.84  # Weight on luminance channel (from Rosenholtz et al.)
    _WGHT_CHROM = 0.08  # Weight on each chrominance channel (from Rosenholtz et al.)
    _WOR = 4  # Number of orientations for subband decomposition
    _ZERO_THRESHOLD = 0.008  # Threshold to consider an array as zeros

    @classmethod
    def _band_entropy(cls, map_: np.ndarray) -> List[float]:
        """
        Compute Shannon entropies of all the subbands.
        
        Args:
            map_: A monochromatic image
            
        Returns:
            A list containing Shannon entropies of all the subbands
        """
        import pyrtools as pt
        
        # Decompose the image into subbands using Steerable Pyramid
        SFpyr = pt.pyramids.SteerablePyramidFreq(
            map_, height=cls._W_LEVELS, order=cls._WOR - 1
        )
        S = SFpyr.pyr_coeffs
        
        en_band = []
        for ind in S.keys():
            subband = S[ind].ravel()
            # Number of bins = sqrt(number of coefficients) as per Rosenholtz et al.
            n_bins = max(1, int(np.sqrt(len(subband))))
            en_band.append(entropy(subband, bins=n_bins))
        
        return en_band

    @classmethod
    def execute(cls, img: Image.Image) -> Dict[str, float]:
        """
        Execute the subband entropy metric.
        
        Args:
            img: PIL Image
            
        Returns:
            Dictionary with subband_entropy value and component values
        """
        try:
            import pyrtools as pt
        except ImportError:
            return {"error": "pyrtools not available", "subband_entropy": None}
        
        from concurrent.futures import ThreadPoolExecutor
        
        # Convert to RGB
        img_rgb = img.convert("RGB")
        img_rgb_nparray = np.array(img_rgb)
        
        # Convert to LAB color space using skimage for proper ranges
        try:
            from skimage import color
            # skimage gives: L: 0-100, a: -128 to 127, b: -128 to 127
            lab = color.rgb2lab(img_rgb_nparray, illuminant="D65", observer="2")
        except ImportError:
            # Fallback to OpenCV with proper range conversion
            img_bgr = cv2.cvtColor(img_rgb_nparray, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
            # OpenCV LAB: L: 0-255, a: 0-255, b: 0-255
            # Convert to standard: L: 0-100, a: -128 to 127, b: -128 to 127
            lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
            lab[:, :, 1] = lab[:, :, 1] - 128.0
            lab[:, :, 2] = lab[:, :, 2] - 128.0
        
        lab_float = lab.astype(np.float64)
        
        # Split into L, a, b channels
        L = lab_float[:, :, 0]  # Luminance: 0-100
        a = lab_float[:, :, 1]  # Chrominance a: -128 to 127
        b = lab_float[:, :, 2]  # Chrominance b: -128 to 127
        
        # Normalize channels to roughly [0, 1] range for steerable pyramid
        # This is critical for proper entropy calculation
        L_norm = L / 100.0  # Normalize L to [0, 1]
        a_norm = (a + 128.0) / 255.0  # Normalize a to [0, 1]
        b_norm = (b + 128.0) / 255.0  # Normalize b to [0, 1]
        
        # Preprocess chrominance channels - check if there's any variation
        if np.max(a_norm) - np.min(a_norm) < cls._ZERO_THRESHOLD:
            a_norm = np.zeros_like(a_norm)
        if np.max(b_norm) - np.min(b_norm) < cls._ZERO_THRESHOLD:
            b_norm = np.zeros_like(b_norm)
        
        # Compute subband entropy for all channels in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_L = executor.submit(cls._band_entropy, L_norm)
            future_a = executor.submit(cls._band_entropy, a_norm)
            future_b = executor.submit(cls._band_entropy, b_norm)
            
            en_band_L = future_L.result()
            en_band_a = future_a.result()
            en_band_b = future_b.result()
        
        # Sum entropies across all subbands (not mean) as per Rosenholtz et al.
        # The measure represents total bits required for subband coding
        luminance_entropy = float(np.sum(en_band_L))
        chrom_a_entropy = float(np.sum(en_band_a))
        chrom_b_entropy = float(np.sum(en_band_b))
        
        # IMPORTANT: The raw sum can be very large (e.g., 75+ bits for 13+ subbands).
        # The paper's "sum" refers to combining channels, not necessarily raw summation.
        # To get scientifically meaningful values (0-8 bits range), we need to normalize.
        # Common approach: divide by number of subbands to get average entropy per subband
        n_subbands_L = len(en_band_L)
        n_subbands_a = len(en_band_a)
        n_subbands_b = len(en_band_b)
        
        # Average entropy per subband (in bits)
        avg_luminance_entropy = luminance_entropy / n_subbands_L if n_subbands_L > 0 else 0.0
        avg_chrom_a_entropy = chrom_a_entropy / n_subbands_a if n_subbands_a > 0 else 0.0
        avg_chrom_b_entropy = chrom_b_entropy / n_subbands_b if n_subbands_b > 0 else 0.0
        
        # Combine entropies with official weights from the paper:
        # 0.84 for luminance, 0.08 for each chrominance channel
        clutter_se = cls._WGHT_LUM * avg_luminance_entropy + cls._WGHT_CHROM * avg_chrom_a_entropy + cls._WGHT_CHROM * avg_chrom_b_entropy
        
        return {
            "subband_entropy": clutter_se,
            "luminance_entropy": avg_luminance_entropy,
            "chrom_a_entropy": avg_chrom_a_entropy,
            "chrom_b_entropy": avg_chrom_b_entropy,
            "luminance_entropy_total": luminance_entropy,
            "chrom_a_entropy_total": chrom_a_entropy,
            "chrom_b_entropy_total": chrom_b_entropy,
            "n_subbands": {"L": n_subbands_L, "a": n_subbands_a, "b": n_subbands_b},
        }
    
    @classmethod
    def compute_score(
        cls,
        entropy_mean: float,
        params: Optional[ScoreParameters] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute subband entropy score component using Gaussian normalization.
        
        Based on main.py formula:
        entropy_score = weight * score * exp(-((mean - target_mean)^2) / (2 * target_std^2))
        
        Args:
            entropy_mean: Mean subband entropy value across slides
            params: Score parameters (uses default if None)
            
        Returns:
            Tuple of (total_score, component_dict)
        """
        if params is None:
            params = get_default_parameters()
        
        # Gaussian for entropy
        entropy_gaussian = math.exp(
            -((entropy_mean - params.w_subband_entropy_mean) ** 2) / (2 * params.w_subband_entropy_std ** 2)
        )
        
        entropy_score = params.weight_subband_entropy * params.weight_subband_entropy_score * entropy_gaussian
        
        return entropy_score, {
            "entropy_total": entropy_score,
            "entropy_gaussian_normalized": entropy_gaussian,
        }


class FeatureCongestionMetric:
    """
    Metric: Feature Congestion.
    
    Measures visual clutter based on color, contrast, and orientation congestion.
    Based on Rosenholtz et al.'s work on measuring visual clutter.
    Requires pyrtools library (macOS/Linux only).
    """
    
    # Constants
    _NUM_LEVELS = 3
    _CONTRAST_FILT_SIGMA = 1
    _CONTRAST_POOL_SIGMA = 3
    _COLOR_POOL_SIGMA = 3
    _ORIENT_POOL_SIGMA = 7 / 2
    _ORIENT_NOISE = 0.001
    _OPP_ENERGY_NOISE = 1.0
    _OPP_ENERGY_FILTER_SCALE = 16 / 14 * 1.75
    _OPP_ENERGY_POOL_SCALE = 1.75
    _COLOR_COEF = 0.2088
    _CONTRAST_COEF = 0.0660
    _ORIENT_COEF = 0.0269
    _MINKOWSKI_ORDER = 1.0
    
    # Class-level pyramid storage
    _L_pyr = {}
    _a_pyr = {}
    _b_pyr = {}

    @staticmethod
    def _collapse(clutter_levels: List) -> np.ndarray:
        """Collapse over scales by taking the maximum (vectorized)."""
        try:
            import pyrtools as pt
        except ImportError:
            raise ImportError("pyrtools is required for feature congestion metric")
        
        kernel_1d = np.array([[0.05, 0.25, 0.4, 0.25, 0.05]])
        kernel_2d = conv2(kernel_1d, kernel_1d.T)
        
        clutter_map = clutter_levels[0].copy()
        for scale in range(1, len(clutter_levels)):
            clutter_here = clutter_levels[scale]
            for kk in range(scale, 0, -1):
                clutter_here = pt.upConv(
                    image=clutter_here,
                    filt=kernel_2d,
                    edge_type="reflect1",
                    step=[2, 2],
                    start=[0, 0],
                )
            # Use vectorized numpy maximum instead of nested loops
            common_sz = (min(clutter_map.shape[0], clutter_here.shape[0]),
                        min(clutter_map.shape[1], clutter_here.shape[1]))
            clutter_map[:common_sz[0], :common_sz[1]] = np.maximum(
                clutter_map[:common_sz[0], :common_sz[1]],
                clutter_here[:common_sz[0], :common_sz[1]]
            )
        return clutter_map

    @classmethod
    def _color_clutter(cls) -> np.ndarray:
        """Compute the color clutter map."""
        covMx = {}
        color_clutter_levels = [0] * cls._NUM_LEVELS
        DL = [0] * cls._NUM_LEVELS
        Da = [0] * cls._NUM_LEVELS
        Db = [0] * cls._NUM_LEVELS
        
        deltaL2 = 0.0007**2
        deltaa2 = 0.1**2
        deltab2 = 0.05**2
        
        bigG = RRgaussfilter1D(round(2 * cls._COLOR_POOL_SIGMA), cls._COLOR_POOL_SIGMA)
        
        for i in range(cls._NUM_LEVELS):
            DL[i] = RRoverlapconv(bigG, cls._L_pyr[(i, 0)])
            DL[i] = RRoverlapconv(bigG.T, DL[i])
            Da[i] = RRoverlapconv(bigG, cls._a_pyr[(i, 0)])
            Da[i] = RRoverlapconv(bigG.T, Da[i])
            Db[i] = RRoverlapconv(bigG, cls._b_pyr[(i, 0)])
            Db[i] = RRoverlapconv(bigG.T, Db[i])
            
            covMx[(i, 0, 0)] = RRoverlapconv(bigG, cls._L_pyr[(i, 0)] ** 2)
            covMx[(i, 0, 0)] = RRoverlapconv(bigG.T, covMx[(i, 0, 0)]) - DL[i] ** 2 + deltaL2
            covMx[(i, 0, 1)] = RRoverlapconv(bigG, cls._L_pyr[(i, 0)] * cls._a_pyr[(i, 0)])
            covMx[(i, 0, 1)] = RRoverlapconv(bigG.T, covMx[(i, 0, 1)]) - DL[i] * Da[i]
            covMx[(i, 0, 2)] = RRoverlapconv(bigG, cls._L_pyr[(i, 0)] * cls._b_pyr[(i, 0)])
            covMx[(i, 0, 2)] = RRoverlapconv(bigG.T, covMx[(i, 0, 2)]) - DL[i] * Db[i]
            covMx[(i, 1, 1)] = RRoverlapconv(bigG, cls._a_pyr[(i, 0)] ** 2)
            covMx[(i, 1, 1)] = RRoverlapconv(bigG.T, covMx[(i, 1, 1)]) - Da[i] ** 2 + deltaa2
            covMx[(i, 1, 2)] = RRoverlapconv(bigG, cls._a_pyr[(i, 0)] * cls._b_pyr[(i, 0)])
            covMx[(i, 1, 2)] = RRoverlapconv(bigG.T, covMx[(i, 1, 2)]) - Da[i] * Db[i]
            covMx[(i, 2, 2)] = RRoverlapconv(bigG, cls._b_pyr[(i, 0)] ** 2)
            covMx[(i, 2, 2)] = RRoverlapconv(bigG.T, covMx[(i, 2, 2)]) - Db[i] ** 2 + deltab2
            
            detIm = (
                covMx[(i, 0, 0)] * (covMx[(i, 1, 1)] * covMx[(i, 2, 2)] - covMx[(i, 1, 2)] * covMx[(i, 1, 2)])
                - covMx[(i, 0, 1)] * (covMx[(i, 0, 1)] * covMx[(i, 2, 2)] - covMx[(i, 1, 2)] * covMx[(i, 0, 2)])
                + covMx[(i, 0, 2)] * (covMx[(i, 0, 1)] * covMx[(i, 1, 2)] - covMx[(i, 1, 1)] * covMx[(i, 0, 2)])
            )
            color_clutter_levels[i] = np.sqrt(np.abs(detIm)) ** (1 / 3)
        
        return cls._collapse(color_clutter_levels)

    @classmethod
    def _contrast_clutter(cls) -> np.ndarray:
        """Compute the contrast clutter map."""
        contrast = RRcontrast1channel(cls._L_pyr, 1)
        contrast_clutter_levels = [0] * cls._NUM_LEVELS
        bigG = RRgaussfilter1D(round(6), 3)
        
        for scale in range(cls._NUM_LEVELS):
            meanD = RRoverlapconv(bigG, contrast[scale])
            meanD = RRoverlapconv(bigG.T, meanD)
            meanD2 = RRoverlapconv(bigG, contrast[scale] ** 2)
            meanD2 = RRoverlapconv(bigG.T, meanD2)
            stddevD = np.sqrt(np.abs(meanD2 - meanD**2))
            contrast_clutter_levels[scale] = stddevD
        
        return cls._collapse(contrast_clutter_levels)

    @classmethod
    def _rr_orientation_opp_energy(cls) -> list:
        """Compute oriented opponent energy."""
        hvdd = [0] * cls._NUM_LEVELS
        hv = [0] * cls._NUM_LEVELS
        dd = [0] * cls._NUM_LEVELS
        out = [0] * cls._NUM_LEVELS
        total = [0] * cls._NUM_LEVELS
        
        for scale in range(cls._NUM_LEVELS):
            hvdd[scale] = orient_filtnew(cls._L_pyr[(scale, 0)], cls._OPP_ENERGY_FILTER_SCALE)
            hvdd[scale] = [x**2 for x in hvdd[scale]]
            hvdd[scale] = poolnew(hvdd[scale], cls._OPP_ENERGY_POOL_SCALE)
            hv[scale] = HV(hvdd[scale])
            dd[scale] = DD(hvdd[scale])
            total[scale] = sumorients(hvdd[scale]) + cls._OPP_ENERGY_NOISE
            hv[scale] = hv[scale] / total[scale]
            dd[scale] = dd[scale] / total[scale]
            out[scale] = (hv[scale], dd[scale])
        return out

    @classmethod
    def _orientation_clutter(cls) -> np.ndarray:
        """Compute the orientation clutter map."""
        Dc = [0] * cls._NUM_LEVELS
        Ds = [0] * cls._NUM_LEVELS
        angles = cls._rr_orientation_opp_energy()
        bigG = RRgaussfilter1D(round(8 * cls._ORIENT_POOL_SIGMA), 4 * cls._ORIENT_POOL_SIGMA)
        
        covMx = {}
        orientation_clutter_levels = [0] * cls._NUM_LEVELS
        
        for i in range(cls._NUM_LEVELS):
            cmx = angles[i][0]
            smx = angles[i][1]
            
            Dc[i] = RRoverlapconv(bigG, cmx)
            Dc[i] = RRoverlapconv(bigG.T, Dc[i])
            Ds[i] = RRoverlapconv(bigG, smx)
            Ds[i] = RRoverlapconv(bigG.T, Ds[i])
            
            covMx[(i, 0, 0)] = RRoverlapconv(bigG, cmx**2)
            covMx[(i, 0, 0)] = RRoverlapconv(bigG.T, covMx[(i, 0, 0)]) - Dc[i] ** 2 + cls._ORIENT_NOISE
            covMx[(i, 0, 1)] = RRoverlapconv(bigG, cmx * smx)
            covMx[(i, 0, 1)] = RRoverlapconv(bigG.T, covMx[(i, 0, 1)]) - Dc[i] * Ds[i]
            covMx[(i, 1, 1)] = RRoverlapconv(bigG, smx**2)
            covMx[(i, 1, 1)] = RRoverlapconv(bigG.T, covMx[(i, 1, 1)]) - Ds[i] ** 2 + cls._ORIENT_NOISE
            
            detIm = covMx[(i, 0, 0)] * covMx[(i, 1, 1)] - covMx[(i, 0, 1)] ** 2
            orientation_clutter_levels[i] = np.abs(detIm) ** (1 / 4)
        
        return cls._collapse(orientation_clutter_levels)

    @classmethod
    def execute(cls, img: Image.Image, max_pixels: int = 5000000) -> Dict[str, Any]:
        """
        Execute the feature congestion metric.
        
        Args:
            img: PIL Image
            max_pixels: Maximum pixels to process (image will be resized if larger)
            
        Returns:
            Dictionary with feature_congestion score and component scores
        """
        try:
            import pyrtools as pt
        except ImportError:
            return {"error": "pyrtools not available", "feature_congestion": None}
        
        from concurrent.futures import ThreadPoolExecutor
        
        # Resize image if too large to reduce computation
        img_rgb = img.convert("RGB")
        width, height = img_rgb.size
        n_pixels = width * height
        
        if n_pixels > max_pixels:
            scale = (max_pixels / n_pixels) ** 0.5
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = img_rgb.resize((new_width, new_height), Image.LANCZOS)
        
        img_rgb_nparray = np.array(img_rgb)
        
        # Convert to LAB
        lab = rgb2lab_clutter(img_rgb_nparray)
        lab_float = lab.astype(np.float32)
        
        L = lab_float[:, :, 0]
        a = lab_float[:, :, 1]
        b = lab_float[:, :, 2]
        
        # Build Gaussian pyramids
        pyr = pt.pyramids.GaussianPyramid(L, height=cls._NUM_LEVELS)
        cls._L_pyr = pyr.pyr_coeffs
        pyr = pt.pyramids.GaussianPyramid(a, height=cls._NUM_LEVELS)
        cls._a_pyr = pyr.pyr_coeffs
        pyr = pt.pyramids.GaussianPyramid(b, height=cls._NUM_LEVELS)
        cls._b_pyr = pyr.pyr_coeffs
        
        # Compute clutter maps in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            color_future = executor.submit(cls._color_clutter)
            contrast_future = executor.submit(cls._contrast_clutter)
            orientation_future = executor.submit(cls._orientation_clutter)
            
            color_clutter_map = color_future.result()
            contrast_clutter_map = contrast_future.result()
            orientation_clutter_map = orientation_future.result()
        
        # Combine clutter maps
        clutter_map_fc = (
            color_clutter_map / cls._COLOR_COEF
            + contrast_clutter_map / cls._CONTRAST_COEF
            + orientation_clutter_map / cls._ORIENT_COEF
        )
        
        # Compute scalar measure
        clutter_scalar_fc = float(
            np.mean(clutter_map_fc**cls._MINKOWSKI_ORDER) ** (1 / cls._MINKOWSKI_ORDER)
        )
        
        return {
            "feature_congestion": clutter_scalar_fc,
            "color_clutter": float(np.mean(color_clutter_map)),
            "contrast_clutter": float(np.mean(contrast_clutter_map)),
            "orientation_clutter": float(np.mean(orientation_clutter_map)),
        }


class VisualHRVMetric:
    """Presentation-level pacing metric based on RMSSD over per-slide scores.

    Given a sequence of normalized slide scores S_1..S_n in [0,1], compute:
    - Flux: Δ_i = |S_{i+1} - S_i|
    - Visual HRV (RMSSD): sqrt(mean(Δ_i^2))
    - Overload count: moving average window=3 exceeds threshold (default 0.75)
    - Final score: 100 * (1 - |(RMSSD - 0.25)/0.25|) - 10 * overload_count
    """

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))
    
    @classmethod
    def calculate_rmssd(
        cls,
        raw_scores: List[float],
        params: Optional[ScoreParameters] = None
    ) -> float:
        """
        Calculate RMSSD from raw scores using sigmoid normalization.
        
        Based on main.py formula:
        - Apply sigmoid: S(x) = 1 / (1 + exp(-k * (x - mu)))
        - Calculate deltas: Δ_i = |S_{i+1} - S_i|
        - RMSSD = sqrt(mean(Δ_i^2))
        
        Args:
            raw_scores: List of raw subband entropy values
            params: Score parameters (uses default if None)
            
        Returns:
            RMSSD value
        """
        if params is None:
            params = get_default_parameters()
        
        if len(raw_scores) < 2:
            return 0.0
        
        # Apply sigmoid normalization
        normalized = []
        for raw_score in raw_scores:
            score = 1 / (1 + math.exp(-params.se_k * (raw_score - params.se_mu)))
            normalized.append(score)
        
        # Calculate deltas
        deltas = np.abs(np.diff(normalized))
        
        # Calculate RMSSD
        rmssd = float(np.sqrt(np.mean(deltas ** 2)))
        return rmssd
    
    @classmethod
    def compute_score(
        cls,
        raw_scores: List[float],
        params: Optional[ScoreParameters] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute RMSSD score component from raw subband entropy scores.
        
        Based on main.py formula:
        - Calculate RMSSD using sigmoid-normalized scores
        - rmssd_score = weight * 100 * (1 - min(1, |rmssd - target| / halfwidth))
        
        Args:
            raw_scores: List of raw subband entropy values from each slide
            params: Score parameters (uses default if None)
            
        Returns:
            Tuple of (total_score, component_dict)
        """
        if params is None:
            params = get_default_parameters()
        
        if not raw_scores or len(raw_scores) < 2:
            return 0.0, {
                "rmssd_total": 0.0,
                "rmssd_value": 0.0,
                "normalized_scores": [],
            }
        
        # Apply sigmoid normalization
        normalized = []
        for raw_score in raw_scores:
            score = 1 / (1 + math.exp(-params.se_k * (raw_score - params.se_mu)))
            normalized.append(score)
        
        # Calculate RMSSD
        rmssd = cls.calculate_rmssd(raw_scores, params)
        
        # RMSSD score calculation (from main.py)
        rmssd_diff = abs(rmssd - params.weight_target_rmssd)
        if params.weight_target_halfwidth > 0:
            rmssd_score = params.weight_subband_entropy * 100.0 * (
                1.0 - min(1.0, rmssd_diff / params.weight_target_halfwidth)
            )
        else:
            rmssd_score = 0.0
        
        return rmssd_score, {
            "rmssd_total": rmssd_score,
            "rmssd_value": rmssd,
            "rmssd_diff_from_target": rmssd_diff,
            "normalized_scores": normalized,
        }

    @classmethod
    def compute(
        cls,
        scores: List[float],
        slide_indices: Optional[List[int]] = None,
        overload_window: int = 3,
        overload_threshold: float = 0.75,
        target_rmssd: float = 0.25,
        target_halfwidth: float = 0.25,
        overload_penalty: float = 10.0,
    ) -> Dict[str, Any]:
        """Compute Visual HRV pacing metrics from per-slide scores.

        Args:
            scores: Per-slide normalized scores in [0,1]
            overload_window: Moving average window length (default 3)
            overload_threshold: Overload threshold for moving average (default 0.75)
            target_rmssd: Ideal RMSSD center (default 0.25)
            target_halfwidth: Half-width for ideal band (default 0.25)
            overload_penalty: Points subtracted per overload event (default 10)

        Returns:
            Dict with rmssd, overload_count, final_score_0_100, and interpretation.
        """
        if not scores:
            return {
                "rmssd": None,
                "overload_count": 0,
                "deltas": [],
                "delta_pairs": [],
                "final_score": None,
                "interpretation": "insufficient_data",
                "n_slides_used": 0,
            }

        s = np.array([cls._clamp01(float(v)) for v in scores], dtype=np.float64)
        if slide_indices is None:
            slide_indices = list(range(int(s.size)))
        if len(slide_indices) != int(s.size):
            raise ValueError("slide_indices must match scores length")

        # RMSSD over absolute successive differences
        if s.size < 2:
            rmssd = 0.0
            deltas = np.array([], dtype=np.float64)
        else:
            deltas = np.abs(np.diff(s))
            rmssd = float(np.sqrt(np.mean(deltas**2)))

        # Overload events via moving average
        overload_count = 0
        overload_indices: List[int] = []
        if overload_window >= 1 and s.size >= overload_window:
            kernel = np.ones(overload_window, dtype=np.float64) / float(overload_window)
            moving_avg = np.convolve(s, kernel, mode="valid")
            overload_mask = moving_avg > float(overload_threshold)
            overload_count = int(np.sum(overload_mask))
            overload_indices = [int(i) for i in np.where(overload_mask)[0].tolist()]

        # Scoring formula
        if target_halfwidth <= 0:
            base_score = 0.0
        else:
            base_score = 100.0 * (1.0 - abs((rmssd - float(target_rmssd)) / float(target_halfwidth)))

        final_score = base_score - (float(overload_penalty) * overload_count)
        final_score = float(max(0.0, min(100.0, final_score)))

        # Interpretation bands (based on RMSSD)
        if rmssd < 0.1:
            interpretation = "flatline"
        elif rmssd > 0.5:
            interpretation = "strobe_light"
        elif 0.15 <= rmssd <= 0.35:
            interpretation = "healthy_pulse"
        else:
            interpretation = "transitional"

        return {
            "rmssd": rmssd,
            "overload_count": overload_count,
            "overload_window": int(overload_window),
            "overload_threshold": float(overload_threshold),
            "overload_window_start_indices": overload_indices,
            "deltas": [float(x) for x in deltas.tolist()],
            "final_score": final_score,
            "interpretation": interpretation,
            "n_slides_used": int(s.size),
        }


# ----------------------------------------------------------------------------
# Total Aesthetics Score Calculator
# ----------------------------------------------------------------------------

def calculate_total_aesthetics_score(
    metrics: Dict[str, Any],
    params: Optional[ScoreParameters] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate total aesthetics score from aggregated metrics using configurable parameters.
    
    This function mirrors the calculation in main.py's calculate_aesthetics_score function.
    
    Args:
        metrics: Dictionary containing aggregated metrics from SlideAestheticsCalculator.
                 Expected keys:
                 - figure_ground_contrast: {"mean": float}
                 - colorfulness: {"mean": float, "std": float}
                 - color_harmony: {"raw_distances": List[float]} or {"deck_metrics": {...}}
                 - subband_entropy: {"mean": float}
                 - visual_hrv: {"normalization": {"raw_scores": List[float]}} or {"rmssd": float}
        params: Score parameters (uses default if None)
        
    Returns:
        Tuple of (total_score, score_components_dict)
    """
    if params is None:
        params = get_default_parameters()
    
    components = {}
    
    # 1. Figure-ground contrast component
    fg_contrast = metrics.get("figure_ground_contrast", {})
    contrast_mean = fg_contrast.get("mean", 0.5)
    contrast_score, _ = FigureGroundContrastMetric.compute_score(contrast_mean, params)
    components["contrast"] = contrast_score
    
    # 2. Colorfulness component
    colorfulness = metrics.get("colorfulness", {})
    color_mean = colorfulness.get("mean", 50.0)
    color_std = colorfulness.get("std", 10.0)
    colorfulness_score, _ = ColorfulnessMetric.compute_score(color_mean, color_std, params)
    components["colorfulness"] = colorfulness_score
    
    # 3. Color harmony component
    harmony = metrics.get("color_harmony", {})
    raw_distances = harmony.get("raw_distances", [])
    
    if raw_distances:
        harmony_score, _ = ColorHarmonyMetric.compute_score(raw_distances, params)
    else:
        # Fallback to pre-computed deck_metrics if available
        deck_metrics = harmony.get("deck_metrics", {})
        mean_score = deck_metrics.get("mean_score", 0.0)
        deck_consistency = deck_metrics.get("deck_consistency", 0.0)
        harmony_score = params.weight_harmony * (
            params.weight_harmony_mean * mean_score -
            params.weight_harmony_deck * deck_consistency
        )
    components["harmony"] = harmony_score
    
    # 4. Subband entropy component
    entropy = metrics.get("subband_entropy", {})
    entropy_mean = entropy.get("mean", 3.5)
    entropy_score, _ = SubbandEntropyMetric.compute_score(entropy_mean, params)
    components["entropy"] = entropy_score
    
    # 5. Visual HRV (RMSSD) component
    visual_hrv = metrics.get("visual_hrv", {})
    normalization = visual_hrv.get("normalization", {})
    raw_scores = normalization.get("raw_scores", [])
    
    if raw_scores:
        rmssd_score, _ = VisualHRVMetric.compute_score(raw_scores, params)
    else:
        # Fallback to pre-computed rmssd
        rmssd = visual_hrv.get("rmssd", 0.1)
        rmssd_diff = abs(rmssd - params.weight_target_rmssd)
        if params.weight_target_halfwidth > 0:
            rmssd_score = params.weight_subband_entropy * 100.0 * (
                1.0 - min(1.0, rmssd_diff / params.weight_target_halfwidth)
            )
        else:
            rmssd_score = 0.0
    components["rmssd"] = rmssd_score
    
    # Total score
    total_score = sum(components.values())
    
    return total_score, components


# ----------------------------------------------------------------------------
# Main Aesthetics Calculator Class
# ----------------------------------------------------------------------------

class SlideAestheticsCalculator:
    """
    Calculator for computing aesthetic metrics on PowerPoint slide images.
    """

    def __init__(self):
        """
        Initialize the calculator.
        """
        pass

        self._all_metrics = {
            "figure_ground_contrast",
            "lab",
            "hsv",
            "color_harmony",
            "visual_complexity",
            "colorfulness",
            "subband_entropy",
        }

    @staticmethod
    def _parse_metrics_list(metrics_to_compute: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        if metrics_to_compute is None:
            return None
        if isinstance(metrics_to_compute, str):
            parts = [p.strip() for p in metrics_to_compute.split(",") if p.strip()]
            return parts
        return list(metrics_to_compute)

    def calculate_slide_metrics(
        self,
        image_path: str,
        metrics_to_compute: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all aesthetic metrics for a single slide image.
        
        Args:
            image_path: Path to the slide image
            
        Returns:
            Dictionary with all calculated metrics
        """
        try:
            img = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return {"error": str(e)}
        
        metrics_list = self._parse_metrics_list(metrics_to_compute)
        selected = None
        if metrics_list is not None:
            selected = set(metrics_list)
            unknown = selected - self._all_metrics
            if unknown:
                logger.warning(f"Unknown metrics requested: {sorted(unknown)}")
            selected = selected & self._all_metrics

        def _want(name: str) -> bool:
            return selected is None or name in selected

        metrics: Dict[str, Any] = {}
        timing = {}
        
        # Figure-ground contrast
        if _want("figure_ground_contrast"):
            try:
                start = time.time()
                metrics["figure_ground_contrast"] = FigureGroundContrastMetric.execute(img, image_path=image_path)
                timing["figure_ground_contrast"] = time.time() - start
                logger.info(f"FigureGroundContrast took {timing['figure_ground_contrast']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate figure-ground contrast: {e}")
                metrics["figure_ground_contrast"] = None
        
        # LAB statistics
        if _want("lab"):
            try:
                start = time.time()
                metrics["lab"] = LABMetric.execute(img)
                timing["lab"] = time.time() - start
                # logger.info(f"LAB took {timing['lab']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate LAB metrics: {e}")
                metrics["lab"] = None
        
        # HSV statistics
        if _want("hsv"):
            try:
                start = time.time()
                metrics["hsv"] = HSVMetric.execute(img)
                timing["hsv"] = time.time() - start
                # logger.info(f"HSV took {timing['hsv']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate HSV metrics: {e}")
                metrics["hsv"] = None
        
        # Color harmony (parallel version for better performance)
        if _want("color_harmony"):
            try:
                start = time.time()
                metrics["color_harmony"] = ColorHarmonyMetric.execute(img)
                timing["color_harmony"] = time.time() - start
                # logger.info(f"ColorHarmony took {timing['color_harmony']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate color harmony: {e}")
                metrics["color_harmony"] = None
        
        # Visual complexity
        if _want("visual_complexity"):
            try:
                start = time.time()
                metrics["visual_complexity"] = VisualComplexityMetric.execute(img)
                timing["visual_complexity"] = time.time() - start
                # logger.info(f"VisualComplexity took {timing['visual_complexity']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate visual complexity: {e}")
                metrics["visual_complexity"] = None
        
        # Colorfulness
        if _want("colorfulness"):
            try:
                start = time.time()
                metrics["colorfulness"] = ColorfulnessMetric.execute(img)
                timing["colorfulness"] = time.time() - start
                # logger.info(f"Colorfulness took {timing['colorfulness']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate colorfulness: {e}")
                metrics["colorfulness"] = None
        
        # Subband Entropy (expensive metric - requires pyrtools)
        if _want("subband_entropy"):
            try:
                start = time.time()
                metrics["subband_entropy"] = SubbandEntropyMetric.execute(img)
                timing["subband_entropy"] = time.time() - start
                # logger.info(f"SubbandEntropy took {timing['subband_entropy']:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to calculate subband entropy: {e}")
                metrics["subband_entropy"] = None
        
        # Log total time and timing breakdown
        total_time = sum(timing.values())
        logger.info(f"Total metrics calculation: {total_time:.3f}s")
        metrics["_timing"] = timing

        if selected is not None:
            metrics["_metrics_requested"] = sorted(list(set(metrics_list or [])))
            metrics["_metrics_selected"] = sorted(list(selected))
        
        return metrics

    def _calculate_slide_metrics_with_index(self, args: Tuple[int, str, Optional[Union[str, List[str]]]]) -> Tuple[int, Dict[str, Any]]:
        """
        Helper method to calculate metrics for a single slide with its index.
        
        Args:
            args: Tuple of (slide_index, image_path)
            
        Returns:
            Tuple of (slide_index, metrics_dict)
        """
        idx, path, metrics_to_compute = args
        metrics = self.calculate_slide_metrics(path, metrics_to_compute=metrics_to_compute)
        metrics["slide_index"] = idx
        metrics["slide_path"] = path
        return (idx, metrics)

    def calculate_presentation_metrics(
        self, 
        image_paths: Union[str, List[str]],
        aggregate: bool = True,
        n_workers: int = 4,
        parallel: bool = True,
        metrics_to_compute: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate aesthetic metrics for all slides in a presentation.
        
        Args:
            image_paths: Either a directory path containing images, or a list of image paths.
                         If a directory, will search for common image formats (png, jpg, jpeg, bmp, gif).
            aggregate: If True, also compute aggregated statistics
            n_workers: Number of parallel workers (default: 4)
            parallel: If True, process slides in parallel (default: True)
            
        Returns:
            Dictionary with per-slide metrics and optionally aggregated stats
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        import glob
        
        # Handle directory input
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                # Search for image files in the directory
                image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.PNG', '*.JPG', '*.JPEG']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(image_paths, ext)))
                # Sort for consistent ordering
                image_paths = sorted(image_files)
                logger.info(f"Found {len(image_paths)} images in directory")
                if not image_paths:
                    return {"error": "No images found in directory", "num_slides": 0, "slides": []}
            else:
                # Single image path provided as string
                image_paths = [image_paths]
        
        num_slides = len(image_paths)
        
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), num_slides)
        
        slide_metrics = [None] * num_slides  # Pre-allocate to maintain order
        
        if parallel and num_slides > 1 and n_workers > 1:
            # Use ThreadPoolExecutor for parallel processing
            # (ProcessPoolExecutor has issues with class methods and some dependencies)
            logger.info(f"Processing {num_slides} slides in parallel with {n_workers} workers")
            
            try:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all tasks
                    future_to_idx = {}
                    for i, path in enumerate(image_paths):
                        future = executor.submit(self._calculate_slide_metrics_with_index, (i, path, metrics_to_compute))
                        future_to_idx[future] = i
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_idx):
                        try:
                            idx, metrics = future.result()
                            slide_metrics[idx] = metrics
                            logger.debug(f"Completed slide {idx+1}/{num_slides}")
                        except Exception as e:
                            idx = future_to_idx[future]
                            logger.error(f"Failed to process slide {idx}: {e}")
                            slide_metrics[idx] = {
                                "error": str(e),
                                "slide_index": idx,
                                "slide_path": image_paths[idx]
                            }
            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
                # Fallback to sequential processing
                for i, path in enumerate(image_paths):
                    logger.debug(f"Processing slide {i+1}/{num_slides}: {path}")
                    metrics = self.calculate_slide_metrics(path, metrics_to_compute=metrics_to_compute)
                    metrics["slide_index"] = i
                    metrics["slide_path"] = path
                    slide_metrics[i] = metrics
        else:
            # Sequential processing
            for i, path in enumerate(image_paths):
                logger.debug(f"Processing slide {i+1}/{num_slides}: {path}")
                metrics = self.calculate_slide_metrics(path, metrics_to_compute=metrics_to_compute)
                metrics["slide_index"] = i
                metrics["slide_path"] = path
                slide_metrics[i] = metrics
        
        result = {
            "num_slides": num_slides,
            "slides": slide_metrics,
        }
        
        if aggregate and slide_metrics:
            result["aggregated"] = self._aggregate_metrics(slide_metrics)
        
        return result

    def _aggregate_metrics(self, slide_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics across all slides.
        
        Args:
            slide_metrics: List of per-slide metrics
            
        Returns:
            Dictionary with aggregated statistics
        """
        aggregated = {}
        
        # Figure-ground contrast
        # This metric returns a dict with {min, max, mean} per slide
        # We need to extract the mean value from each slide's result
        fgc_mean_values = []
        fgc_min_values = []
        fgc_max_values = []
        for m in slide_metrics:
            fgc = m.get("figure_ground_contrast")
            if fgc is not None:
                if isinstance(fgc, dict):
                    # Extract values from the dict
                    if "mean" in fgc:
                        fgc_mean_values.append(fgc["mean"])
                    if "min" in fgc:
                        fgc_min_values.append(fgc["min"])
                    if "max" in fgc:
                        fgc_max_values.append(fgc["max"])
                else:
                    # Fallback for scalar values (old format)
                    fgc_mean_values.append(fgc)
        
        if fgc_mean_values:
            aggregated["figure_ground_contrast"] = {
                "mean": float(np.mean(fgc_mean_values)),
                "std": float(np.std(fgc_mean_values)),
                "min": float(np.min(fgc_mean_values)),
                "max": float(np.max(fgc_mean_values)),
            }
            # Add per-slide statistics if available
            if fgc_min_values:
                aggregated["figure_ground_contrast"]["slide_min_mean"] = float(np.mean(fgc_min_values))
            if fgc_max_values:
                aggregated["figure_ground_contrast"]["slide_max_mean"] = float(np.mean(fgc_max_values))
        
        # LAB averages
        lab_keys = ["L_avg", "L_std", "A_avg", "A_std", "B_avg", "B_std"]
        lab_values = {k: [] for k in lab_keys}
        for m in slide_metrics:
            lab = m.get("lab")
            if lab:
                for k in lab_keys:
                    if k in lab:
                        lab_values[k].append(lab[k])
        
        aggregated["lab"] = {}
        for k, values in lab_values.items():
            if values:
                aggregated["lab"][k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        
        # HSV averages
        hsv_keys = ["H_avg", "S_avg", "S_std", "V_avg", "V_std"]
        hsv_values = {k: [] for k in hsv_keys}
        for m in slide_metrics:
            hsv = m.get("hsv")
            if hsv:
                for k in hsv_keys:
                    if k in hsv:
                        hsv_values[k].append(hsv[k])
        
        aggregated["hsv"] = {}
        for k, values in hsv_values.items():
            if values:
                aggregated["hsv"][k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        
        # Color harmony
        harmony_distances = [m["color_harmony"]["best_distance"] 
                            for m in slide_metrics 
                            if m.get("color_harmony") and "best_distance" in m["color_harmony"]]
        if harmony_distances:
            aggregated["color_harmony"] = {
                "mean_distance": float(np.mean(harmony_distances)),
                "std_distance": float(np.std(harmony_distances)),
                "min_distance": float(np.min(harmony_distances)),
                "max_distance": float(np.max(harmony_distances)),
            }
            
            # Get current parameters for harmony sigma
            params = get_default_parameters()
            
            # Compute deck-level metrics using configurable sigma
            slide_scores = []
            for distance in harmony_distances:
                score = ColorHarmonyMetric.compute_slide_score_with_params(distance, params)
                slide_scores.append(score)
            
            deck_metrics = ColorHarmonyMetric.compute_deck_metrics(slide_scores)
            aggregated["color_harmony"]["deck_metrics"] = deck_metrics
            
            # Store raw distances for each slide
            aggregated["color_harmony"]["raw_distances"] = harmony_distances
        
        # Visual complexity
        complexity_values = []
        edge_density_values = []
        for m in slide_metrics:
            vc = m.get("visual_complexity")
            if vc:
                if "complexity_score" in vc:
                    complexity_values.append(vc["complexity_score"])
                if "edge_density" in vc:
                    edge_density_values.append(vc["edge_density"])
        
        if complexity_values:
            aggregated["visual_complexity"] = {
                "mean_complexity": float(np.mean(complexity_values)),
                "std_complexity": float(np.std(complexity_values)),
                "mean_edge_density": float(np.mean(edge_density_values)) if edge_density_values else None,
            }
        
        # Colorfulness
        colorfulness_values = [m["colorfulness"] for m in slide_metrics 
                              if m.get("colorfulness") is not None]
        if colorfulness_values:
            # Basic statistics
            std_colorfulness = float(np.std(colorfulness_values))
            
            # Colorfulness pacing score (Plateau effect normalization)
            # Ideal variation is around 15, with a good zone width of 10
            mu_target = 8.0  # Ideal colorfulness variation
            w = 5.0          # Width of "good" zone
            
            # Gaussian normalization: S_norm = exp(-((σ - μ_target)^2) / (2*w^2))
            pacing_score = float(np.exp(-((std_colorfulness - mu_target) ** 2) / (2 * w ** 2)))
            
            aggregated["colorfulness"] = {
                "mean": float(np.mean(colorfulness_values)),
                "std": std_colorfulness,
                "min": float(np.min(colorfulness_values)),
                "max": float(np.max(colorfulness_values)),
                "pacing_score": pacing_score,
                "pacing_normalization": {
                    "method": "gaussian",
                    "formula": "S_norm = exp(-((σ - μ_target)^2) / (2*w^2))",
                    "mu_target": mu_target,
                    "w": w,
                    "sigma_pacing": std_colorfulness,
                }
            }
        
        # Subband Entropy
        se_values = []
        se_luminance_values = []
        for m in slide_metrics:
            se = m.get("subband_entropy")
            if se and "subband_entropy" in se:
                v = se.get("subband_entropy")
                # Filter out None and non-numeric values
                if v is not None:
                    try:
                        se_values.append(float(v))
                    except Exception:
                        # skip non-convertible values
                        continue
            if se and "luminance_entropy" in se:
                lv = se.get("luminance_entropy")
                if lv is not None:
                    try:
                        se_luminance_values.append(float(lv))
                    except Exception:
                        continue
        
        if se_values:
            aggregated["subband_entropy"] = {
                "mean": float(np.mean(se_values)),
                "std": float(np.std(se_values)),
                "min": float(np.min(se_values)),
                "max": float(np.max(se_values)),
            }
            if se_luminance_values:
                aggregated["subband_entropy"]["mean_luminance"] = float(np.mean(se_luminance_values))
        
        # Feature Congestion
        fc_clutter_values = []
        fc_color_values = []
        fc_orientation_values = []
        fc_luminance_values = []
        for m in slide_metrics:
            fc = m.get("feature_congestion")
            if fc:
                if "clutter" in fc:
                    fc_clutter_values.append(fc["clutter"])
                if "color_congestion" in fc:
                    fc_color_values.append(fc["color_congestion"])
                if "orientation_congestion" in fc:
                    fc_orientation_values.append(fc["orientation_congestion"])
                if "luminance_congestion" in fc:
                    fc_luminance_values.append(fc["luminance_congestion"])
        
        if fc_clutter_values:
            aggregated["feature_congestion"] = {
                "mean_clutter": float(np.mean(fc_clutter_values)),
                "std_clutter": float(np.std(fc_clutter_values)),
                "min_clutter": float(np.min(fc_clutter_values)),
                "max_clutter": float(np.max(fc_clutter_values)),
            }
            if fc_color_values:
                aggregated["feature_congestion"]["mean_color_congestion"] = float(np.mean(fc_color_values))
            if fc_orientation_values:
                aggregated["feature_congestion"]["mean_orientation_congestion"] = float(np.mean(fc_orientation_values))
            if fc_luminance_values:
                aggregated["feature_congestion"]["mean_luminance_congestion"] = float(np.mean(fc_luminance_values))
        
        # UMSI (Saliency)
        umsi_fixations = []
        umsi_entropy = []
        for m in slide_metrics:
            umsi = m.get("umsi")
            if umsi:
                if "predicted_fixations" in umsi:
                    umsi_fixations.append(umsi["predicted_fixations"])
                if "saliency_entropy" in umsi:
                    umsi_entropy.append(umsi["saliency_entropy"])
        
        if umsi_fixations:
            aggregated["umsi"] = {
                "mean_predicted_fixations": float(np.mean(umsi_fixations)),
                "std_predicted_fixations": float(np.std(umsi_fixations)),
                "min_predicted_fixations": float(np.min(umsi_fixations)),
                "max_predicted_fixations": float(np.max(umsi_fixations)),
            }
            if umsi_entropy:
                aggregated["umsi"]["mean_saliency_entropy"] = float(np.mean(umsi_entropy))
                aggregated["umsi"]["std_saliency_entropy"] = float(np.std(umsi_entropy))

        # ------------------------------------------------------------------
        # Visual HRV (RMSSD pacing) over per-slide scores S_i in [0,1]
        # ------------------------------------------------------------------
        # Prefer subband_entropy, but normalize to [0,1] using sigmoid
        slide_scores: List[float] = []
        slide_scores_raw: List[float] = []
        slide_score_source: List[str] = []
        slide_score_indices: List[int] = []
        
        # Get sigmoid normalization parameters from config
        params = get_default_parameters()
        se_mu = params.se_mu  # Midpoint for standard business slide
        se_k = params.se_k    # Sensitivity parameter
        
        def normalize_entropy_to_score(x: float, mu: float = se_mu, k: float = se_k) -> float:
            """
            Normalize raw entropy value to [0,1] score using sigmoid function.
            
            Formula: S(x) = 1 / (1 + e^(-k * (x - mu)))
            
            Args:
                x: Raw subband entropy value (in bits)
                mu: Midpoint - entropy of a standard business slide (default 4.25)
                k: Slope/sensitivity parameter (default 1.5)
                
            Returns:
                Normalized score in [0, 1]
            """
            return float(1.0 / (1.0 + np.exp(-k * (x - mu))))

        for m in slide_metrics:
            idx = m.get("slide_index")
            se = m.get("subband_entropy")
            if se and isinstance(se, dict) and se.get("subband_entropy") is not None:
                try:
                    val = float(se["subband_entropy"])
                    if np.isfinite(val):
                        # Store raw value
                        slide_scores_raw.append(val)
                        # Normalize to [0,1] using sigmoid
                        normalized_score = normalize_entropy_to_score(val, mu=se_mu, k=se_k)
                        slide_scores.append(normalized_score)
                        slide_score_source.append("subband_entropy")
                        slide_score_indices.append(int(idx) if idx is not None else len(slide_score_indices))
                        continue
                except Exception:
                    pass

        if slide_scores:
            aggregated["visual_hrv"] = VisualHRVMetric.compute(slide_scores, slide_indices=slide_score_indices)
            # Add normalization metadata
            aggregated["visual_hrv"]["normalization"] = {
                "method": "sigmoid",
                "formula": "S(x) = 1 / (1 + exp(-k * (x - mu)))",
                "mu": se_mu,
                "k": se_k,
                "raw_scores": slide_scores_raw,
                "normalized_scores": slide_scores,
            }
        
        return aggregated


def main():
    """Test the aesthetics metrics on a sample image."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Calculate slide aesthetics metrics")
    parser.add_argument("input", nargs="+", help="Path(s) to image file(s) or a directory containing images")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file path. If not specified, prints to stdout.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--workers", "-w", type=int, default=2,
                       help="Number of parallel workers (default: 2)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated list of metrics to compute per slide. "
            "Example: lab,hsv,visual_complexity,subband_entropy. "
            "If omitted, computes all available metrics."
        ),
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to JSON config file for score parameters. See ScoreParameters for available options.",
    )
    parser.add_argument(
        "--compute-score",
        action="store_true",
        help="Compute total aesthetics score from aggregated metrics (requires --config or uses defaults).",
    )
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load score parameters from config file if provided
    params = None
    if args.config:
        try:
            params = load_parameters_from_config(args.config)
            logger.info(f"Loaded score parameters from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config file {args.config}: {e}")
            return
    
    calculator = SlideAestheticsCalculator()
    
    # Determine input type
    if len(args.input) == 1:
        input_path = args.input[0]
        if os.path.isdir(input_path):
            # Directory input - process all images in directory
            metrics = calculator.calculate_presentation_metrics(
                input_path,
                n_workers=args.workers,
                parallel=not args.no_parallel,
                metrics_to_compute=args.metrics,
            )
        elif os.path.isfile(input_path):
            # Single file input
            metrics = calculator.calculate_slide_metrics(input_path, metrics_to_compute=args.metrics)
        else:
            logger.error(f"Input path does not exist: {input_path}")
            return
    else:
        # Multiple file inputs
        metrics = calculator.calculate_presentation_metrics(
            args.input,
            n_workers=args.workers,
            parallel=not args.no_parallel,
            metrics_to_compute=args.metrics,
        )
    
    # Compute total aesthetics score if requested
    if args.compute_score and "aggregated" in metrics:
        total_score, score_components = calculate_total_aesthetics_score(
            metrics["aggregated"],
            params=params
        )
        metrics["aesthetics_score"] = {
            "total": total_score,
            "components": score_components,
            "parameters_used": params.to_dict() if params else get_default_parameters().to_dict(),
        }
        logger.info(f"Total aesthetics score: {total_score:.2f}")
    
    # Output results
    if args.output:
        # Write to JSON file
        output_path = args.output
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
        print(f"Results saved to {output_path}")
    else:
        # Print to stdout
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
