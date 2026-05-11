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