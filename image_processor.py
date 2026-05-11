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