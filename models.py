"""
models.py - Data models and base classes for the Spot the Difference game.
================ 
Chepngenoh owns this file.
 
Responsibilities:
  - DifferenceRegion: stores position/size/type/found state for each difference
  - GameState: tracks score, mistakes, found differences, game phase
  - BaseAlteration: abstract base class for image alteration strategies (polymorphism)
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import abc

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class GamePhase(Enum):
    """Represents the current phase of a game round."""
    WAITING   = auto()   # No image loaded yet
    PLAYING   = auto()   # Player is actively finding differences
    COMPLETED = auto()   # All 5 differences found
    LOCKED    = auto()   # 3 mistakes reached – input disabled
    REVEALED  = auto()   # Reveal button pressed

class AlterationType(Enum):
    """Supported image alteration types."""
    COLOUR_SHIFT   = "Colour Shift"
    BLUR           = "Blur"
    BRIGHTNESS     = "Brightness"
    NOISE          = "Noise"
    PIXELATE       = "Pixelate"
    INVERT         = "Invert"
    SHARPEN        = "Sharpen"

# ---------------------------------------------------------------------------
# DifferenceRegion
# ---------------------------------------------------------------------------

@dataclass
class DifferenceRegion:
    """
    Represents a single altered region (difference) in the modified image.

    Attributes:
        x, y        : Top-left corner of the region (in original image pixels).
        width       : Width of the region in pixels.
        height      : Height of the region in pixels.
        alt_type    : The type of alteration applied.
        found       : Whether the player has found this difference.
        revealed    : Whether the reveal button exposed this difference.
    """
    x: int
    y: int
    width: int
    height: int
    alt_type: AlterationType
    found: bool = False
    revealed: bool = False

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains_point(self, px: int, py: int) -> bool:
        """Return True if (px, py) lies within this region."""
        return self.x <= px <= self.right and self.y <= py <= self.bottom

    def within_tolerance(self, px: int, py: int, tolerance: int = 30) -> bool:
        """
        Return True if (px, py) is within `tolerance` pixels of the region
        boundary or interior.  This gives the player a fair hitbox.
        """
        expanded_x1 = self.x - tolerance
        expanded_y1 = self.y - tolerance
        expanded_x2 = self.right + tolerance
        expanded_y2 = self.bottom + tolerance
        return expanded_x1 <= px <= expanded_x2 and expanded_y1 <= py <= expanded_y2

    def overlaps(self, other: "DifferenceRegion", padding: int = 10) -> bool:
        """Return True if this region overlaps `other` (with optional padding)."""
        return not (
            self.right  + padding <= other.x or
            other.right + padding <= self.x  or
            self.bottom + padding <= other.y or
            other.bottom + padding <= self.y
        )

    def __repr__(self) -> str:
        return (
            f"DifferenceRegion(x={self.x}, y={self.y}, "
            f"w={self.width}, h={self.height}, "
            f"type={self.alt_type.value}, found={self.found})"
        )


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------
class GameState:
    """
    Encapsulates the mutable state for a single game session.

    Encapsulation: all mutation goes through methods so invariants are kept.
    """

    MAX_MISTAKES   = 3
    TOTAL_DIFFS    = 5

    def __init__(self):
        self._phase:       GamePhase                    = GamePhase.WAITING
        self._differences: List[DifferenceRegion]       = []
        self._mistakes:    int                          = 0
        self._total_found: int                          = 0   # cumulative across images
        self._round_found: int                          = 0   # for the current image

    # ------------------------------------------------------------------
    # Properties (read-only outside)
    # ------------------------------------------------------------------

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def differences(self) -> List[DifferenceRegion]:
        return list(self._differences)          # return a copy

    @property
    def mistakes(self) -> int:
        return self._mistakes

    @property
    def total_found(self) -> int:
        return self._total_found

    @property
    def round_found(self) -> int:
        return self._round_found

    @property
    def remaining(self) -> int:
        return self.TOTAL_DIFFS - self._round_found

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def new_round(self, differences: List[DifferenceRegion]) -> None:
        """Start a new round with freshly generated differences."""
        if len(differences) != self.TOTAL_DIFFS:
            raise ValueError(
                f"Expected {self.TOTAL_DIFFS} differences, got {len(differences)}."
            )
        self._differences = differences
        self._mistakes    = 0
        self._round_found = 0
        self._phase       = GamePhase.PLAYING

    def register_hit(self, region: DifferenceRegion) -> None:
        """Mark a difference as found."""
        if self._phase != GamePhase.PLAYING:
            return
        if region.found:
            return
        region.found = True
        self._round_found  += 1
        self._total_found  += 1
        if self._round_found == self.TOTAL_DIFFS:
            self._phase = GamePhase.COMPLETED

    def register_mistake(self) -> bool:
        """
        Record a wrong click.
        Returns True if the game should now be locked (3 mistakes reached).
        """
        if self._phase != GamePhase.PLAYING:
            return False
        self._mistakes += 1
        if self._mistakes >= self.MAX_MISTAKES:
            self._phase = GamePhase.LOCKED
            return True
        return False

    def reveal_all(self) -> List[DifferenceRegion]:
        """Reveal remaining unfound differences; return them."""
        unfound = [d for d in self._differences if not d.found]
        for d in unfound:
            d.revealed = True
        self._phase = GamePhase.REVEALED
        return unfound

    def find_matching_region(
        self,
        px: int,
        py: int,
        tolerance: int = 30
    ) -> Optional[DifferenceRegion]:
        """
        Return the first unfound difference that contains (px, py) within
        tolerance, or None.
        """
        for region in self._differences:
            if not region.found and region.within_tolerance(px, py, tolerance):
                return region
        return None

    def __repr__(self) -> str:
        return (
            f"GameState(phase={self._phase.name}, "
            f"mistakes={self._mistakes}, "
            f"round_found={self._round_found}, "
            f"total_found={self._total_found})"
        )


# ---------------------------------------------------------------------------
# BaseAlteration – abstract strategy (polymorphism / inheritance)
# ---------------------------------------------------------------------------

class BaseAlteration(abc.ABC):
    """
    Abstract base class for image alteration strategies.

    Subclasses must implement `apply(image, region)` which modifies the
    numpy array in-place (or returns a new one) for the given region.
    """

    @property
    @abc.abstractmethod
    def alteration_type(self) -> AlterationType:
        """Return the enum value identifying this alteration."""

    @abc.abstractmethod
    def apply(self, image, region: DifferenceRegion):
        """
        Apply the alteration to `image` within `region`.

        Parameters
        ----------
        image   : numpy.ndarray  (BGR, uint8)
        region  : DifferenceRegion

        Returns
        -------
        numpy.ndarray  – modified image (may be same object as input)
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.alteration_type.value})"
