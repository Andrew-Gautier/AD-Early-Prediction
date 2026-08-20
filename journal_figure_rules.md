Based on the PLOS Digital Health figure requirements, here is a comprehensive prompt you can use as a guideline for creating publication-ready TIFF figures in your Python research workflow.

---

## PROMPT: PLOS Digital Health–Compliant TIFF Figure Generation in Python

### Objective
Generate a TIFF figure file that meets all PLOS Digital Health technical specifications for main article figures, using Python (Matplotlib, Pillow, or similar libraries).

### Figure File Requirements Checklist

#### 1. File Format
- **Format:** TIFF only (EPS is also accepted but TIFF is strongly preferred)
- **Naming:** Save as `Fig1.tif`, `Fig2.tif`, etc., matching the in-text citation and caption label

#### 2. Dimensions
- **Minimum width:** 789 pixels (at 300 dpi)
- **Maximum width:** 2250 pixels (at 300 dpi)
- **Maximum height:** 2625 pixels (at 300 dpi)
- *Note:* Dimensions refer to the entire figure, excluding white space
- *Tip:* For alignment with the text column, make width no wider than 5.2 inches (13.2 cm)

#### 3. Resolution
- **Range:** 300–600 dpi
- **Critical:** Do not increase resolution beyond the original image's native pixel count—this creates artificial data and misrepresents the original
- Resolution below 300 dpi results in pixelated figures; above 600 dpi may lead to unwanted resizing

#### 4. File Size
- **Maximum:** 10 MB or less

#### 5. Color Mode
- **Allowed:** RGB (8 bit/channel) or grayscale only

#### 6. Compression (Required)
- **Use LZW compression**
- **Flatten:** No layers—flatten the image to a single background layer

#### 7. Alpha Channels
- **No alpha channels** allowed

#### 8. Fonts
- **Allowed fonts:** Arial, Times, or Symbol only
- **Font size:** 8–12 point
- **Do NOT include** author names, article title, or figure number/title/caption within the figure file—these go in the manuscript caption

#### 9. Multi-panel Figures
- Place all panels from a multi-part figure into **a single page and single file**
- Use presentation software to combine panels, then convert to TIFF
- **Do NOT drag/drop or copy/paste** images—this results in 72 dpi resolution
- If your figure has numerous small elements, render at **600 dpi**

#### 10. White Space and Borders
- **Recommended:** 2-point white space border around each figure to prevent cropping
- **Crop out** excess white space around image content

#### 11. Orientation
- Submit the figure in the **final desired orientation**—it will be published as supplied

#### 12. Reporting Precision
- Round summary statistics shown in figure legends/labels (e.g., mean AUC/AP and their std or CI spread) to the **nearest 0.005**, using a single shared helper (`round_to_increment` in `visualization.py`) rather than plain `.3f`/`.2f` string formatting, which only rounds to the nearest 0.001/0.01 and can silently diverge from the intended precision.
- Keep saved data files (e.g. `aggregate_stats.csv`) at full float precision—only round at display/formatting time, so every reporting surface (figure legend, table, printed summary) stays consistent even if the values are computed by different code paths.

---

### Python Implementation Guidelines

When generating your figure in Python, use the following code patterns:

#### Matplotlib Example

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Set up figure with correct dimensions
# For a full-width figure at 300 dpi: 2250 pixels wide
# For a half-column figure: ~789 pixels wide
dpi = 300
width_inches = 7.5  # 2250/300
height_inches = 8.75  # 2625/300 (max)

fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)

# --- Your plotting code here ---
# Use Arial, Times, or Symbol fonts
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10  # 8-12 point range

# --- Save as TIFF with LZW compression ---
# Note: Matplotlib's savefig does not support LZW compression directly.
# Save as a temporary file, then use PIL to apply LZW compression.

temp_path = 'temp_figure.png'
fig.savefig(temp_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)  # 2pt ≈ 0.02in at 300dpi

# Open with PIL and save as TIFF with LZW compression
from PIL import Image
img = Image.open(temp_path)
# Convert to RGB (8 bit/channel) if not already
if img.mode != 'RGB':
    img = img.convert('RGB')
# Save as TIFF with LZW compression
img.save('Fig1.tif', format='TIFF', compression='tiff_lzw')

# Clean up
import os
os.remove(temp_path)
```

#### Pillow (PIL) Direct Approach

If you're assembling figures from multiple images:

```python
from PIL import Image

# Open images, combine into a single multi-panel figure
# Ensure all images are at 300-600 dpi
# ...

# Save final figure
final_image.save('Fig1.tif', 
                  format='TIFF', 
                  compression='tiff_lzw',
                  dpi=(300, 300))
```

### Final Checklist Before Saving

- [ ] Format: TIFF (.tif)
- [ ] Width: 789–2250 pixels at 300 dpi
- [ ] Height: ≤ 2625 pixels at 300 dpi
- [ ] Resolution: 300–600 dpi
- [ ] File size: < 10 MB
- [ ] Color mode: RGB (8 bit/channel) or grayscale
- [ ] Compression: LZW
- [ ] Flattened: No layers
- [ ] No alpha channels
- [ ] Fonts: Arial, Times, or Symbol only (8–12 pt)
- [ ] No figure number/title/caption inside the file
- [ ] File name: Fig1.tif, Fig2.tif, etc.
- [ ] 2-point white space border (recommended)
- [ ] Excess white space cropped
- [ ] Correct orientation

### Additional Resources

- PLOS provides a free tool called **NAAS** to check and convert figures
- For blot/gel images, provide original uncropped images as a separate supporting information file (PDF or TIFF with LZW compression)

---

This prompt can be directly integrated into your research workflow as a checklist and code template for generating PLOS Digital Health–compliant TIFF figures in Python.