"""
game_logic.py – Game controller: ties together GameState, ImageProcessor,
and drives round transitions.
================ 
Rakebun owns this file.
 
Responsibilities:
  - GameController: single façade used by the GUI
  - Exposes clean methods: load_image, register_click, reveal, reset_round
  - Raises typed exceptions so the GUI can give helpful messages
"""

import os
from typing import List, Optional, Tuple, Callable

from models import GameState, DifferenceRegion, GamePhase
from image_processor import ImageProcessor


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""


class GameNotReadyError(Exception):
    """Raised when a game action is attempted before an image is loaded."""


# ---------------------------------------------------------------------------
# ClickResult
# ---------------------------------------------------------------------------

class ClickResult:
    """Value object returned by GameController.register_click()."""

    def __init__(
        self,
        hit:        bool,
        region:     Optional[DifferenceRegion],
        locked:     bool,
        completed:  bool,
        mistakes:   int,
        remaining:  int,
    ):
        self.hit       = hit        # True  → correct click
        self.region    = region     # The matched region (or None)
        self.locked    = locked     # Game just became locked
        self.completed = completed  # All differences found
        self.mistakes  = mistakes
        self.remaining = remaining

    def __repr__(self) -> str:
        return (
            f"ClickResult(hit={self.hit}, locked={self.locked}, "
            f"completed={self.completed}, mistakes={self.mistakes}, "
            f"remaining={self.remaining})"
        )


# ---------------------------------------------------------------------------
# GameController
# ---------------------------------------------------------------------------

class GameController:
    """
    Façade / controller for the Spot the Difference game.

    Coordinates:
      - ImageProcessor  (image loading, cloning, drawing)
      - GameState       (score, mistakes, phase)

    The GUI only interacts with GameController; it never calls
    ImageProcessor or GameState directly.
    """

    # Tolerance in pixels for a click to register as a hit
    CLICK_TOLERANCE = 35

    def __init__(self):
        self._processor   = ImageProcessor()
        self._state       = GameState()
        self._image_path: Optional[str] = None

        # Cached display copies (with circles drawn)
        self._display_orig: Optional = None
        self._display_mod:  Optional = None

        # Optional callbacks so the GUI can subscribe to events
        self._on_state_change: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def set_state_change_callback(self, cb: Callable) -> None:
        """GUI can register a callback invoked on every state change."""
        self._on_state_change = cb

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def phase(self) -> GamePhase:
        return self._state.phase

    @property
    def display_original(self):
        return self._display_orig

    @property
    def display_modified(self):
        return self._display_mod

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------

    def load_image(self, filepath: str) -> None:
        """
        Load a new image, generate differences, and start a new round.
        Raises ImageLoadError on failure.
        """
        if not os.path.isfile(filepath):
            raise ImageLoadError(f"File not found: '{filepath}'")

        try:
            self._processor.load_image(filepath)
            regions = self._processor.generate_differences()
        except (ValueError, IOError, RuntimeError) as exc:
            raise ImageLoadError(str(exc)) from exc

        self._image_path    = filepath
        self._state.new_round(regions)
        self._refresh_display_copies()
        self._notify()

    def register_click(self, display_x: int, display_y: int) -> ClickResult:
        """
        Handle a click on the modified image at (display_x, display_y).
        Returns a ClickResult with all relevant state info.
        Raises GameNotReadyError if no image has been loaded.
        """
        if self._state.phase not in (GamePhase.PLAYING,):
            raise GameNotReadyError(
                "Cannot register click: game is not in PLAYING phase "
                f"(current: {self._state.phase.name})."
            )

        img_x, img_y = self._processor.map_display_to_image_coords(display_x, display_y)
        region = self._state.find_matching_region(img_x, img_y, self.CLICK_TOLERANCE)

        locked    = False
        completed = False

        if region is not None:
            self._state.register_hit(region)
            # Draw red circle on both display copies
            RED = (0, 0, 255)
            self._processor.draw_circle_on_image(self._display_orig, region, RED)
            self._processor.draw_circle_on_image(self._display_mod,  region, RED)
            completed = self._state.phase == GamePhase.COMPLETED
        else:
            locked = self._state.register_mistake()

        result = ClickResult(
            hit       = region is not None,
            region    = region,
            locked    = locked,
            completed = completed,
            mistakes  = self._state.mistakes,
            remaining = self._state.remaining,
        )
        self._notify()
        return result

    def reveal(self) -> List[DifferenceRegion]:
        """
        Reveal all unfound differences with blue circles.
        Returns the list of revealed regions.
        Raises GameNotReadyError if no image loaded.
        """
        if self._state.phase == GamePhase.WAITING:
            raise GameNotReadyError("No image loaded.")

        revealed = self._state.reveal_all()
        BLUE = (255, 100, 0)
        for region in revealed:
            self._processor.draw_circle_on_image(self._display_orig, region, BLUE)
            self._processor.draw_circle_on_image(self._display_mod,  region, BLUE)

        self._notify()
        return revealed

    def get_summary(self) -> dict:
        """
        Return a dict of current state suitable for the GUI status bar.
        """
        return {
            "phase":       self._state.phase.name,
            "remaining":   self._state.remaining,
            "mistakes":    self._state.mistakes,
            "total_found": self._state.total_found,
            "round_found": self._state.round_found,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_display_copies(self) -> None:
        """Make fresh display copies from the processor's images."""
        import numpy as np
        orig = self._processor.original
        mod  = self._processor.modified
        if orig is not None:
            self._display_orig = orig.copy()
        if mod is not None:
            self._display_mod  = mod.copy()

    def _notify(self) -> None:
        if self._on_state_change:
            try:
                self._on_state_change()
            except Exception:
                pass