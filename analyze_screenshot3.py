#!/usr/bin/env python3
"""Final analysis: precise Claude Code chat input field location."""

from PIL import Image
from collections import defaultdict, Counter

img = Image.open('/root/screenshot_vision.png')
w, h = img.size

print(f"Image: {w}x{h}\n")

# === 1. Find ALL bright rectangles at the bottom of the screen ===
print("=== All bright areas in bottom 200px ===")
bright_regions = []
for y in range(h - 200, h, 2):
    for x in range(0, w, 2):
        r, g, b = img.getpixel((x, y))
        brightness = (r + g + b) / 3
        if brightness > 100:
            bright_regions.append((x, y, brightness, r, g, b))

# Group by y to find horizontal segments
from collections import defaultdict
by_y = defaultdict(list)
for x, y, brightness, r, g, b in bright_regions:
    by_y[y].append((x, brightness))

print(f"Total bright pixels in bottom 200px: {len(bright_regions)}")

# Find the topmost and bottommost bright rows
if by_y:
    print(f"Bright rows range: y={min(by_y.keys())} to y={max(by_y.keys())}")

# For each y, find continuous bright segments
for y in sorted(by_y.keys()):
    x_vals = sorted(by_y[y])
    segments = []
    start = x_vals[0][0]
    prev = start
    for x, b in x_vals[1:]:
        if x - prev > 10:  # gap
            segments.append((start, prev, prev - start))
            start = x
        prev = x
    segments.append((start, prev, prev - start))
    
    # Only print wide segments
    wide = [(s, e, width) for s, e, width in segments if width > 50]
    if wide:
        print(f"  y={y:4d}: {wide}")

# === 2. Find the tallest bright rectangle ===
print("\n=== Finding the brightest/tallest rectangular region ===")
# Scan columns for vertical bright stretches
for x_start in range(0, w, 50):
    x_end = min(x_start + 100, w)
    # For this column range, find vertical bright stretches in bottom 200px
    in_bright = False
    bright_start = 0
    stretches = []
    for y in range(h - 200, h):
        bright_count = 0
        for x in range(x_start, x_end, 2):
            px = img.getpixel((x, y))
            if sum(px)/3 > 80:
                bright_count += 1
        # If more than half the pixels in this band are bright
        if bright_count > (x_end - x_start) // 4:
            if not in_bright:
                bright_start = y
                in_bright = True
        else:
            if in_bright:
                height = y - bright_start
                if height > 15:
                    stretches.append((bright_start, y, height))
                in_bright = False
    if in_bright:
        height = h - bright_start
        if height > 15:
            stretches.append((bright_start, h, height))
    
    if stretches:
        for s, e, height in stretches:
            print(f"  x=[{x_start}-{x_end}]: bright stretch y=[{s}-{e}] height={height}")

# === 3. Precise Claude Code input field detection ===
# The chat input field in Claude Code typically has these characteristics:
# - Located at the BOTTOM of the chat panel
# - Has a text input area (light/white rectangle)
# - Has a send button (often an arrow icon)
# - May have a file attachment button
# - Often has a darker border/background around it

print("\n=== Precise bottom-of-panel analysis ===")
# Look at the very bottom rows (y=1050-1080) for UI elements
# The text input is usually the last interactive element before the window edge

# Scan every pixel row in the last 40px
for y in range(h - 40, h):
    # Find groups of pixels with specific colors
    white_buttons = []
    input_bg = []
    for x in range(0, w, 1):
        r, g, b = img.getpixel((x, y))
        if r > 200 and g > 200 and b > 200:  # near-white
            white_buttons.append(x)
        elif 180 < r < 220 and 180 < g < 220 and 180 < b < 220:  # light gray (input bg)
            input_bg.append(x)
    
    if white_buttons:
        # group consecutive
        groups = []
        start = white_buttons[0]
        prev = start
        for x in white_buttons[1:]:
            if x - prev > 3:
                groups.append((start, prev, prev - start))
                start = x
            prev = x
        groups.append((start, prev, prev - start))
        for s, e, width in groups:
            if width > 50:
                print(f"  y={y}: WHITE button [{s}-{e}] width={width}")

# === 4. Check for specific VS Code panel elements ===
print("\n=== Checking for VS Code panel structure ===")
# VS Code bottom panel has tabs (PROBLEMS, OUTPUT, TERMINAL, etc.)
# Look for text-like patterns in the bottom area

# Sample color around where panel tabs would be
for y_range_name, y_start, y_end in [("Tab area", h-300, h-250), ("Content area", h-250, h-50), ("Input area", h-50, h)]:
    color_counter = Counter()
    for y in range(y_start, y_end, 3):
        for x in range(0, w, 10):
            px = img.getpixel((x, y))
            color_counter[px] += 1
    top = color_counter.most_common(5)
    print(f"  {y_range_name} (y={y_start}-{y_end}): top colors: {top}")

# === 5. Overall layout determination ===
print("\n=== LAYOUT ANALYSIS ===")
# Determine the structure of the visible area
sections = []
current_brightness = None
section_start = 0
for y in range(0, h, 5):
    pixels = [img.getpixel((x, y)) for x in range(0, w, 20)]
    avg = sum(sum(p)/3 for p in pixels) / len(pixels)
    
    if current_brightness is None:
        current_brightness = avg
        section_start = y
    elif abs(avg - current_brightness) > 15:
        sections.append((section_start, y, current_brightness))
        section_start = y
        current_brightness = avg

sections.append((section_start, h, current_brightness))

for s, e, b in sections:
    label = ""
    if b < 5: label = "(very dark)"
    elif b < 30: label = "(dark)"
    elif b < 60: label = "(dim)"
    elif b < 90: label = "(moderate)"
    elif b < 120: label = "(bright)"
    else: label = "(very bright)"
    print(f"  y=[{s:4d}-{e:4d}] ({e-s:3d}px): avg_brightness={b:5.1f} {label}")

# === 6. FINAL: Exact coordinates of the chat input ===
print("\n=== FINAL: Chat Input Field Detection ===")
# Based on the analysis, the input field should be a light/white rectangle
# at the bottom of the Claude Code panel.
# Let's find its exact center coordinates

# The white rectangle we found: x=[480-696], y=[1042-1070]
# Center would be at approximately x=(480+696)/2=588, y=(1042+1070)/2=1056

# But let me verify this is actually a text input by checking its surroundings
print("Checking surroundings of the white rectangle at x=480-696, y=1042-1070:")
# Above the rectangle
for y_check in range(1030, 1045):
    avg_b = sum(sum(img.getpixel((x, y_check))/3) for x in range(480, 696, 5)) / ((696-480)//5)
    print(f"  Above (y={y_check}): avg brightness = {avg_b:.1f}")
    
# Below the rectangle  
for y_check in range(1070, h):
    if y_check < h:
        avg_b = sum(sum(img.getpixel((x, y_check))/3) for x in range(480, 696, 5)) / ((696-480)//5)
        print(f"  Below (y={y_check}): avg brightness = {avg_b:.1f}")

# Check if there's text or a send button nearby
for y_check in range(1040, h):
    for x_check in range(460, 720):
        r, g, b = img.getpixel((x_check, y_check))
        if (r < 50 and g < 50 and b > 180) or (r > 200 and g < 50 and b < 50):
            print(f"  Accent color found at ({x_check}, {y_check}): RGB=({r},{g},{b})")

print("\n=== Sending coordinates ===")
print(f"Input field bounding box: x=[480, 696], y=[1042, 1070]")
center_x = (480 + 696) // 2
center_y = (1042 + 1070) // 2
print(f"Center click point: ({center_x}, {center_y})")