#!/usr/bin/env python3
"""Ultimate analysis: identify VS Code + Claude Code UI in screenshot."""

from PIL import Image
from collections import Counter

img = Image.open('/root/screenshot_vision.png')
w, h = img.size

print(f"Image: {w}x{h}\n")

# === Detect VS Code window boundaries ===
# VS Code on Windows has a distinctive dark title bar, activity bar on left,
# and panel at bottom. Let's find the exact window.

print("=== Full-width average brightness profile ===")
for y in range(0, h, 1):
    pixels = [img.getpixel((x, y)) for x in range(0, w, 10)]
    avg_b = sum(sum(p)/3 for p in pixels) / len(pixels)
    
    # Detect sharp transitions
    if y > 0:
        prev_pixels = [img.getpixel((x, y-1)) for x in range(0, w, 10)]
        prev_avg = sum(sum(p)/3 for p in prev_pixels) / len(prev_pixels)
        diff = abs(avg_b - prev_avg)
        
        if diff > 25:
            print(f"  y={y:4d}: avg_bright={avg_b:5.1f} (Δ={diff:.1f}) <-- TRANSITION")

# === Focus: Find all chat-like input fields ===
# The Claude Code input is typically a <textarea> with:
# - Light background (RGB ~240-255)
# - Rounded corners or border
# - Placeholder text like "Type a message..."
# - Send button (often an arrow icon)
# - Often has a file attachment button

print("\n=== Scanning for potential chat input fields (light rectangles at bottom of panels) ===")

# Scan the entire image for light rectangles that could be input fields
# Criteria: width > 100px, height > 20px, surrounded by darker area

# Find all light horizontal segments
horizontal_segments = {}  # y -> [(x_start, x_end, width)]
for y in range(0, h):
    row_pixels = []
    for x in range(0, w, 2):
        px = img.getpixel((x, y))
        b = sum(px)/3
        row_pixels.append((x, b))
    
    segs = []
    start = None
    last_bright = False
    for x, b in row_pixels:
        if b > 100:
            if not last_bright:
                start = x
                last_bright = True
        else:
            if last_bright:
                width = x - start
                if width > 50:
                    segs.append((start, x, width))
                last_bright = False
    if last_bright:
        width = w - start
        if width > 50:
            segs.append((start, w, width))
    
    if segs:
        horizontal_segments[y] = segs

# Now find rectangles (continuous bright segments across multiple rows)
rectangles = []
processed = set()

for y in sorted(horizontal_segments.keys()):
    for seg in horizontal_segments[y]:
        x_start, x_end, width = seg
        
        # Check how many consecutive rows have a segment at similar x position
        rect_height = 1
        for y2 in range(y+1, min(y+100, h)):
            if y2 in horizontal_segments:
                for seg2 in horizontal_segments[y2]:
                    x2_start, x2_end, w2 = seg2
                    # Check overlap
                    overlap_start = max(x_start, x2_start)
                    overlap_end = min(x_end, x2_end)
                    if overlap_end - overlap_start > min(width, w2) * 0.5:
                        rect_height += 1
                        break
                else:
                    break
            else:
                break
        
        if rect_height > 10 and width > 80:
            key = (x_start, y, x_end, y+rect_height)
            if key not in processed:
                processed.add(key)
                rectangles.append((x_start, y, x_end, y+rect_height, width, rect_height))

# Sort by size (largest area first)
rectangles.sort(key=lambda r: r[4]*r[5], reverse=True)

print(f"Found {len(rectangles)} potential rectangles:")
for x1, y1, x2, y2, width, height in rectangles[:20]:
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    print(f"  Rect: x=[{x1}-{x2}], y=[{y1}-{y2}], size={width}x{height}, center=({center_x},{center_y})")

# === Check the Claude-specific color palette ===
print("\n=== Claude Code brand colors check ===")
# Claude purple: #6C5CE7 (108, 92, 231)
# Claude accent: various purple/violet shades
# Search for purple pixels
purple_pixels = []
for y in range(0, h, 3):
    for x in range(0, w, 3):
        r, g, b = img.getpixel((x, y))
        # Check for purple tones (high R, B but low G relative)
        if b > 150 and r > 50 and g < r + 30 and g < b - 30:
            purple_pixels.append((x, y, r, g, b))

if purple_pixels:
    print(f"Found {len(purple_pixels)} purple-ish pixels")
    # Find the bounding box of purple pixels
    min_x = min(p[0] for p in purple_pixels)
    max_x = max(p[0] for p in purple_pixels)
    min_y = min(p[1] for p in purple_pixels)
    max_y = max(p[1] for p in purple_pixels)
    print(f"  Purple region: x=[{min_x}-{max_x}], y=[{min_y}-{max_y}]")
    # Sample some purple pixels
    for px, py, r, g, b in purple_pixels[:10]:
        print(f"    ({px}, {py}): RGB=({r},{g},{b})")

# === Check for VS Code Activity Bar (left side icons) ===
print("\n=== VS Code Activity Bar analysis (leftmost 70px) ===")
# VS Code activity bar is typically ~50px wide on the left
for y in range(0, h, 10):
    left_colors = [img.getpixel((x, y)) for x in range(0, 70, 5)]
    avg = sum(sum(p)/3 for p in left_colors) / len(left_colors)
    if avg > 5:
        unique_colors = set(left_colors)
        print(f"  y={y}: avg_bright={avg:.1f}, unique_colors={len(unique_colors)}")

# === Check for VS Code side panel ===
print("\n=== Searching for panel borders (vertical lines) ===")
# Vertical borders would show as consistent darker/lighter columns
for x in range(0, w, 50):
    x_range = range(max(0, x-2), min(w, x+3))
    col_brightness = []
    for y in range(0, h, 5):
        pixels = [sum(img.getpixel((cx, y))/3) for cx in x_range]
        col_brightness.append(sum(pixels)/len(pixels))
    avg = sum(col_brightness)/len(col_brightness)
    if avg < 20:  # Very dark vertical line
        # Check if neighbors are brighter
        left_x = max(0, x-15)
        right_x = min(w, x+15)
        left_avg = sum(sum(img.getpixel((left_x, y))/3) for y in range(0, h, 10)) / (h//10)
        right_avg = sum(sum(img.getpixel((right_x, y))/3) for y in range(0, h, 10)) / (h//10)
        if left_avg > 30 or right_avg > 30:
            print(f"  x={x}: avg_bright={avg:.1f}, left(x={left_x})={left_avg:.1f}, right(x={right_x})={right_avg:.1f}  <-- possible border")

# === Look for text "Claude" or chat input placeholder ===
print("\n=== Detecting text areas (small bright marks on dark bg) ===")
# In VS Code dark theme, text is light colored. Let's look for 
# rows of text-like patterns (small groups of light pixels)

# Scan for rows with lots of scattered bright pixels (text)
for y in range(0, h, 5):
    bright_pixels = []
    for x in range(0, w, 2):
        px = img.getpixel((x, y))
        if sum(px)/3 > 150:
            bright_pixels.append(x)
    
    if len(bright_pixels) > 10 and len(bright_pixels) < 500:
        # Group into clusters (words)
        clusters = []
        start = bright_pixels[0]
        prev = start
        for x in bright_pixels[1:]:
            if x - prev > 10:
                clusters.append((start, prev, prev-start))
                start = x
            prev = x
        clusters.append((start, prev, prev-start))
        
        # Filter to text-sized clusters
        text_clusters = [c for c in clusters if 10 < c[2] < 200]
        if 3 <= len(text_clusters) <= 30:
            print(f"  y={y}: {len(text_clusters)} text clusters across row, e.g.: {text_clusters[:5]}")

# === FINAL ANSWER ===
print("\n\n=== FINAL DETERMINATION ===")
print("Based on the comprehensive analysis:")
print()
print("The most significant UI elements at the bottom of the screen are:")
print(f"  1. Bright taskbar-like line at y=950 (x=68-1806, width=1738px)")
print(f"  2. White/light-gray rectangle at y=1041-1070, x=480-695 (center: ~588, 1056)")