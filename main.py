"""
HIT137 Assignment 3 - Spot the Difference Game
Main entry point.

================ 
Tanzim owns this file.
"""
import tkinter as tk
from gui import SpotTheDifferenceApp


def main():
    root = tk.Tk()
    root.title("Spot the Difference")
    root.resizable(True, True)
    app = SpotTheDifferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
