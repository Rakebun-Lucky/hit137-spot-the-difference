"""
gui.py – Tkinter GUI for the Spot the Difference game.
================ 
Saad & Tanzim owns this file.
 
Responsibilities (Saad):
  - SpotTheDifferenceApp: root window orchestrator
  - ImagePanel: reusable canvas panel that displays a single image
  - StatusBar: bottom info strip (remaining, mistakes, total score)
  - ControlPanel: buttons (Load Image, Reveal)
  - DialogHelper: static methods for pop-up dialogs
 
Tanzim wired the click handlers and edge-case feedback into this GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk           # Pillow converts BGR→RGB for Tk

from models import GamePhase
from game_logic import GameController, ImageLoadError, GameNotReadyError


# ---------------------------------------------------------------------------
# ImagePanel
# ---------------------------------------------------------------------------

class ImagePanel(tk.Frame):
    """
    A labelled canvas that shows a single OpenCV image (numpy BGR array).
    Supports click binding (optional).
    """

    PLACEHOLDER_COLOR = "#2b2b2b"
    PLACEHOLDER_TEXT  = "No image loaded"

    def __init__(self, parent, title: str, clickable: bool = False, **kwargs):
        super().__init__(parent, bg="#1e1e1e", **kwargs)
        self._title     = title
        self._clickable = clickable
        self._photo:    Optional[ImageTk.PhotoImage] = None
        self._click_cb  = None

        # Title label
        tk.Label(
            self, text=title, bg="#1e1e1e", fg="#aaaaaa",
            font=("Helvetica", 11, "bold"),
        ).pack(pady=(6, 2))

        # Canvas
        self._canvas = tk.Canvas(
            self,
            bg=self.PLACEHOLDER_COLOR,
            cursor="crosshair" if clickable else "arrow",
            highlightthickness=2,
            highlightbackground="#444444",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self._draw_placeholder()

        if clickable:
            self._canvas.bind("<Button-1>", self._on_canvas_click)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self, bgr_image: np.ndarray) -> None:
        """Show a BGR numpy image on the canvas."""
        rgb   = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(pil)

        h, w = bgr_image.shape[:2]
        self._canvas.config(width=w, height=h)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def clear(self) -> None:
        self._photo = None
        self._canvas.delete("all")
        self._draw_placeholder()

    def set_click_callback(self, cb) -> None:
        self._click_cb = cb

    def set_cursor(self, cursor: str) -> None:
        self._canvas.config(cursor=cursor)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _draw_placeholder(self) -> None:
        self._canvas.config(width=400, height=350)
        self._canvas.create_text(
            200, 175,
            text=self.PLACEHOLDER_TEXT,
            fill="#555555",
            font=("Helvetica", 13),
        )

    def _on_canvas_click(self, event) -> None:
        if self._click_cb:
            self._click_cb(event.x, event.y)


# ---------------------------------------------------------------------------
# StatusBar
# ---------------------------------------------------------------------------

class StatusBar(tk.Frame):
    """Displays game statistics: Remaining, Mistakes, Total Found."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#181818", relief=tk.SUNKEN, **kwargs)

        style = {"bg": "#181818", "fg": "#cccccc", "font": ("Helvetica", 11)}
        accent = {"bg": "#181818", "fg": "#f0a500", "font": ("Helvetica", 11, "bold")}

        # Remaining
        tk.Label(self, text="Remaining: ", **style).grid(row=0, column=0, padx=(14, 0), pady=6)
        self._remaining_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._remaining_var, **accent).grid(row=0, column=1, padx=(0, 20))

        # Mistakes
        tk.Label(self, text="Mistakes: ", **style).grid(row=0, column=2)
        self._mistakes_var = tk.StringVar(value="—")
        tk.Label(self, textvariable=self._mistakes_var, **accent).grid(row=0, column=3, padx=(0, 20))

        # Total found
        tk.Label(self, text="Total Found: ", **style).grid(row=0, column=4)
        self._total_var = tk.StringVar(value="0")
        tk.Label(self, textvariable=self._total_var, **accent).grid(row=0, column=5, padx=(0, 20))

        # Phase
        tk.Label(self, text="Status: ", **style).grid(row=0, column=6)
        self._phase_var = tk.StringVar(value="Waiting for image")
        tk.Label(self, textvariable=self._phase_var, **accent).grid(row=0, column=7, padx=(0, 14))

    def update(self, summary: dict) -> None:
        phase = summary.get("phase", "WAITING")
        self._remaining_var.set(str(summary.get("remaining", "—")))
        self._mistakes_var.set(
            f"{summary.get('mistakes', 0)} / 3"
        )
        self._total_var.set(str(summary.get("total_found", 0)))

        phase_labels = {
            "WAITING":   "Waiting for image",
            "PLAYING":   "🔍 Searching…",
            "COMPLETED": "✅ Round complete!",
            "LOCKED":    "🔒 Too many mistakes",
            "REVEALED":  "👁 Differences revealed",
        }
        self._phase_var.set(phase_labels.get(phase, phase))


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------

class ControlPanel(tk.Frame):
    """Button row: Load Image, Reveal Differences."""

    BTN_STYLE = {
        "font":              ("Helvetica", 11, "bold"),
        "relief":            tk.FLAT,
        "padx":              18,
        "pady":              8,
        "cursor":            "hand2",
        "activeforeground":  "white",
    }

    def __init__(self, parent, on_load, on_reveal, **kwargs):
        super().__init__(parent, bg="#1e1e1e", **kwargs)

        self._load_btn = tk.Button(
            self,
            text="📂  Load Image",
            bg="#3a7bd5",
            fg="white",
            activebackground="#2a5fa5",
            command=on_load,
            **self.BTN_STYLE,
        )
        self._load_btn.pack(side=tk.LEFT, padx=(14, 6), pady=10)

        self._reveal_btn = tk.Button(
            self,
            text="👁  Reveal Differences",
            bg="#c0392b",
            fg="white",
            activebackground="#922b21",
            command=on_reveal,
            state=tk.DISABLED,
            **self.BTN_STYLE,
        )
        self._reveal_btn.pack(side=tk.LEFT, padx=6, pady=10)

        # Instructions label
        tk.Label(
            self,
            text="Click on the RIGHT image to find differences.",
            bg="#1e1e1e",
            fg="#888888",
            font=("Helvetica", 10),
        ).pack(side=tk.RIGHT, padx=14)

    def set_reveal_enabled(self, enabled: bool) -> None:
        self._reveal_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_load_enabled(self, enabled: bool) -> None:
        self._load_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)


# ---------------------------------------------------------------------------
# DialogHelper
# ---------------------------------------------------------------------------

class DialogHelper:
    """Static helpers for pop-up dialogs."""

    @staticmethod
    def info(title: str, message: str) -> None:
        messagebox.showinfo(title, message)

    @staticmethod
    def error(title: str, message: str) -> None:
        messagebox.showerror(title, message)

    @staticmethod
    def warn(title: str, message: str) -> None:
        messagebox.showwarning(title, message)

    @staticmethod
    def ask_open_image() -> Optional[str]:
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG",        "*.jpg *.jpeg"),
                ("PNG",         "*.png"),
                ("BMP",         "*.bmp"),
                ("All files",   "*.*"),
            ],
        )
        return path if path else None


# ---------------------------------------------------------------------------
# SpotTheDifferenceApp  (root window)
# ---------------------------------------------------------------------------

class SpotTheDifferenceApp:
    """
    Main application class.

    Owns:
      - GameController (business logic)
      - ImagePanel × 2 (display)
      - ControlPanel   (buttons)
      - StatusBar      (stats)

    Member 4 is responsible for wiring click handling and all feedback paths.
    """

    APP_BG = "#141414"

    def __init__(self, root: tk.Tk):
        self._root       = root
        self._controller = GameController()
        self._controller.set_state_change_callback(self._on_state_change)

        root.configure(bg=self.APP_BG)
        root.minsize(900, 550)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self._root

        # Title
        tk.Label(
            root,
            text="🔍  Spot the Difference",
            bg=self.APP_BG,
            fg="#f0a500",
            font=("Helvetica", 18, "bold"),
        ).pack(pady=(14, 0))

        # Control panel
        self._ctrl = ControlPanel(
            root,
            on_load=self._cmd_load,
            on_reveal=self._cmd_reveal,
        )
        self._ctrl.pack(fill=tk.X)

        # Image panels in a horizontal row
        img_frame = tk.Frame(root, bg=self.APP_BG)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self._panel_orig = ImagePanel(img_frame, title="Original Image",  clickable=False)
        self._panel_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self._panel_mod  = ImagePanel(img_frame, title="Modified Image ← Click here", clickable=True)
        self._panel_mod.set_click_callback(self._cmd_click)
        self._panel_mod.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # Status bar
        self._status = StatusBar(root)
        self._status.pack(fill=tk.X, side=tk.BOTTOM)

   

   
    # ------------------------------------------------------------------
    # Commands (Tanzim)
    # ------------------------------------------------------------------