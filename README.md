# hit137-spot-the-difference
Tkinter + OpenCV desktop game that generates and detects hidden image differences using object-oriented programming principles.

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
Tkinter is included with standard Python. If missing on Linux:
```bash
sudo apt-get install python3-tk
```

### 2. Run
```bash
python3 main.py
```

## How to Play

1. Click **📂 Load Image** and choose a JPG, PNG, or BMP file.
2. The **left** panel shows the original image; the **right** shows the modified copy with 5 hidden differences.
3. Click on the **right (modified) image** where you spot a difference.
   - ✅ **Correct** → a red circle is drawn around the difference on both images.
   - ❌ **Wrong** → the mistake counter increments.
4. You have **3 mistakes** per round. After 3 mistakes the round locks automatically.
5. Click **👁 Reveal Differences** at any time to expose remaining differences in blue.
6. Load a new image to start the next round. The **Total Found** counter is cumulative across all rounds.
