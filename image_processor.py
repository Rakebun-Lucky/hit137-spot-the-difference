"""
image_processor.py – Image loading, alteration generation and application.
================ 
Rakebun owns this file.

 
Responsibilities:
  - Load & validate images (JPG, PNG, BMP)
  - Scale images preserving aspect ratio
  - Seven concrete alteration strategies (inherits BaseAlteration)
  - DifferenceGenerator: places 5 non-overlapping regions randomly
  - ImageProcessor: orchestrates clone creation + difference application
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Optional

from models import (
    BaseAlteration,
    AlterationType,
    DifferenceRegion,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
NUM_DIFFERENCES      = 5
MIN_REGION_SIZE      = 40    # minimum width/height of a difference region (px)
MAX_REGION_SIZE      = 100   # maximum width/height
OVERLAP_PADDING      = 15    # extra gap between regions
MAX_PLACEMENT_TRIES  = 1000  # give up after this many random attempts


# ---------------------------------------------------------------------------
# Concrete Alteration Strategies
# ---------------------------------------------------------------------------

class ColourShiftAlteration(BaseAlteration):
    """Shift hue/saturation in HSV space – subtle but findable."""

    def __init__(self, hue_shift: int = 20, sat_scale: float = 1.4):
        self._hue_shift  = hue_shift
        self._sat_scale  = sat_scale

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.COLOUR_SHIFT

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        roi = image[y1:y2, x1:x2].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self._hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._sat_scale, 0, 255)
        hsv = hsv.astype(np.uint8)
        image[y1:y2, x1:x2] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image


class BlurAlteration(BaseAlteration):
    """Apply a Gaussian blur to a region."""

    def __init__(self, kernel_size: int = 15):
        # Ensure kernel is odd
        self._ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.BLUR

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        roi = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (self._ksize, self._ksize), 0)
        return image


class BrightnessAlteration(BaseAlteration):
    """Increase or decrease brightness of a region."""

    def __init__(self, delta: int = 60):
        self._delta = delta   # positive = brighter, negative = darker

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.BRIGHTNESS

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        roi = image[y1:y2, x1:x2].astype(np.int32)
        roi = np.clip(roi + self._delta, 0, 255).astype(np.uint8)
        image[y1:y2, x1:x2] = roi
        return image


class NoiseAlteration(BaseAlteration):
    """Add salt-and-pepper noise to a region."""

    def __init__(self, amount: float = 0.08):
        self._amount = amount   # fraction of pixels to corrupt

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.NOISE

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        roi = image[y1:y2, x1:x2].copy()
        h, w = roi.shape[:2]
        num_pixels = int(h * w * self._amount)
        # salt
        for _ in range(num_pixels):
            ry = random.randint(0, h - 1)
            rx = random.randint(0, w - 1)
            roi[ry, rx] = [255, 255, 255]
        # pepper
        for _ in range(num_pixels):
            ry = random.randint(0, h - 1)
            rx = random.randint(0, w - 1)
            roi[ry, rx] = [0, 0, 0]
        image[y1:y2, x1:x2] = roi
        return image


class PixelateAlteration(BaseAlteration):
    """Pixelate (mosaic) a region."""

    def __init__(self, block_size: int = 12):
        self._block = max(block_size, 2)

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.PIXELATE

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        roi = image[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if h < self._block or w < self._block:
            return image
        small = cv2.resize(
            roi,
            (max(1, w // self._block), max(1, h // self._block)),
            interpolation=cv2.INTER_LINEAR,
        )
        image[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return image


class InvertAlteration(BaseAlteration):
    """Invert (negate) the colours of a region."""

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.INVERT

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        image[y1:y2, x1:x2] = cv2.bitwise_not(image[y1:y2, x1:x2])
        return image


class SharpenAlteration(BaseAlteration):
    """Sharpen edges in a region using an unsharp mask."""

    @property
    def alteration_type(self) -> AlterationType:
        return AlterationType.SHARPEN

    def apply(self, image: np.ndarray, region: DifferenceRegion) -> np.ndarray:
        x1, y1, x2, y2 = region.x, region.y, region.right, region.bottom
        roi = image[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (0, 0), 3)
        sharpened = cv2.addWeighted(roi, 2.5, blurred, -1.5, 0)
        image[y1:y2, x1:x2] = np.clip(sharpened, 0, 255).astype(np.uint8)
        return image


# Registry: all available alteration classes
ALL_ALTERATIONS: List[BaseAlteration] = [
    ColourShiftAlteration(),
    BlurAlteration(),
    BrightnessAlteration(delta=60),
    BrightnessAlteration(delta=-60),   # darker variant
    NoiseAlteration(),
    PixelateAlteration(),
    InvertAlteration(),
    SharpenAlteration(),
]


# ---------------------------------------------------------------------------
# DifferenceGenerator
# ---------------------------------------------------------------------------

class DifferenceGenerator:
    """
    Places exactly NUM_DIFFERENCES non-overlapping DifferenceRegion objects
    at random positions within image bounds.

    Uses a retry loop with OVERLAP_PADDING to guarantee separation.
    """

    def __init__(
        self,
        img_width:   int,
        img_height:  int,
        num_diffs:   int  = NUM_DIFFERENCES,
        min_size:    int  = MIN_REGION_SIZE,
        max_size:    int  = MAX_REGION_SIZE,
        padding:     int  = OVERLAP_PADDING,
        max_tries:   int  = MAX_PLACEMENT_TRIES,
    ):
        self._iw        = img_width
        self._ih        = img_height
        self._num       = num_diffs
        self._min       = min_size
        self._max       = max_size
        self._padding   = padding
        self._max_tries = max_tries

    def generate(self) -> List[DifferenceRegion]:
        """
        Return a list of NUM_DIFFERENCES non-overlapping DifferenceRegion objects.
        Alteration types are chosen randomly (without replacement if possible).
        Raises RuntimeError if placement fails after MAX_PLACEMENT_TRIES.
        """
        placed: List[DifferenceRegion] = []
        # Shuffle alteration pool so types are varied
        alt_pool = random.sample(ALL_ALTERATIONS, k=min(self._num, len(ALL_ALTERATIONS)))
        while len(alt_pool) < self._num:
            alt_pool.append(random.choice(ALL_ALTERATIONS))

        for i in range(self._num):
            alteration = alt_pool[i]
            region = self._place_one(placed, alteration)
            placed.append(region)

        return placed

    def _place_one(
        self,
        existing: List[DifferenceRegion],
        alteration: BaseAlteration,
    ) -> DifferenceRegion:
        """Try to place a single region without overlapping existing ones."""
        for _ in range(self._max_tries):
            w = random.randint(self._min, self._max)
            h = random.randint(self._min, self._max)
            # Ensure region fits inside image with a small border
            border = 5
            if self._iw - w - border < border or self._ih - h - border < border:
                continue
            x = random.randint(border, self._iw - w - border)
            y = random.randint(border, self._ih - h - border)
            candidate = DifferenceRegion(
                x=x, y=y, width=w, height=h,
                alt_type=alteration.alteration_type,
            )
            if not any(candidate.overlaps(e, self._padding) for e in existing):
                return candidate

        raise RuntimeError(
            f"Could not place difference region after {self._max_tries} attempts. "
            "The image may be too small."
        )


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------

class ImageProcessor:
    """
    High-level image operations: loading, scaling, cloning, and applying
    differences.

    Class interaction: uses DifferenceGenerator and concrete BaseAlteration
    subclasses to build the modified image.
    """

    # Target display size (max dimension); aspect ratio preserved
    DISPLAY_MAX = 600

    def __init__(self):
        self._original:   Optional[np.ndarray]         = None
        self._modified:   Optional[np.ndarray]         = None
        self._regions:    List[DifferenceRegion]       = []
        self._orig_size:  Tuple[int, int]              = (0, 0)   # (w, h) before scaling
        self._scale:      float                        = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def original(self) -> Optional[np.ndarray]:
        return self._original

    @property
    def modified(self) -> Optional[np.ndarray]:
        return self._modified

    @property
    def regions(self) -> List[DifferenceRegion]:
        return list(self._regions)

    @property
    def scale_factor(self) -> float:
        return self._scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load an image from disk, validate format, and scale it.
        Returns the scaled BGR image, or raises on error.
        """
        # Validate extension
        lower = filepath.lower()
        if not any(lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
            raise ValueError(
                f"Unsupported format. Please use one of: "
                f"{', '.join(SUPPORTED_EXTENSIONS)}"
            )

        img = cv2.imread(filepath)
        if img is None:
            raise IOError(f"Failed to read image: '{filepath}'. "
                          "The file may be corrupt or inaccessible.")

        # Edge case: reject tiny images
        h, w = img.shape[:2]
        min_dim = MIN_REGION_SIZE * 3   # need room for 5 regions
        if h < min_dim or w < min_dim:
            raise ValueError(
                f"Image too small ({w}×{h} px). "
                f"Minimum required: {min_dim}×{min_dim} px."
            )

        self._orig_size = (w, h)
        scaled, self._scale = self._scale_image(img)
        self._original = scaled
        return scaled

    def generate_differences(self) -> List[DifferenceRegion]:
        """
        Clone the loaded image, apply 5 random alterations, and store regions.
        Must be called after load_image().
        """
        if self._original is None:
            raise RuntimeError("No image loaded. Call load_image() first.")

        h, w = self._original.shape[:2]
        generator = DifferenceGenerator(img_width=w, img_height=h)
        regions   = generator.generate()

        # Build alteration map: type → strategy instance
        alt_map = {a.alteration_type: a for a in ALL_ALTERATIONS}

        modified = self._original.copy()
        for region in regions:
            strategy = alt_map.get(region.alt_type)
            if strategy is None:
                raise RuntimeError(f"No strategy for {region.alt_type}")
            strategy.apply(modified, region)

        self._modified = modified
        self._regions  = regions
        return regions

    def draw_circle_on_image(
        self,
        image:   np.ndarray,
        region:  DifferenceRegion,
        colour:  Tuple[int, int, int],
        thickness: int = 3,
    ) -> np.ndarray:
        """
        Draw a circle centred on `region` with the given BGR colour.
        Returns the modified image.
        """
        cx = region.center_x
        cy = region.center_y
        radius = max(region.width, region.height) // 2 + 8
        return cv2.circle(image, (cx, cy), radius, colour, thickness)

    def map_display_to_image_coords(
        self,
        display_x: int,
        display_y: int,
    ) -> Tuple[int, int]:
        """
        Convert coordinates from the displayed (scaled) image back to the
        original image coordinate space.  (Inverse of _scale_image.)
        Useful if scale != 1.0 (currently we work in scaled space, so this
        is an identity transform, but kept for correctness if DISPLAY_MAX changes.)
        """
        return display_x, display_y

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale_image(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Downscale image so its largest dimension is at most DISPLAY_MAX.
        Preserves aspect ratio.  Returns (scaled_image, scale_factor).
        """
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.DISPLAY_MAX:
            return img.copy(), 1.0

        scale  = self.DISPLAY_MAX / max_dim
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scaled, scale