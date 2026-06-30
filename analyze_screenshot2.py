#!/usr/bin/env python3
"""Optimized analysis of VS Code screenshot for Claude Code chat input field."""

from PIL import Image
from collections import Counter

img = Image.open('/root/screenshot_vision.png')
w, h = img.size

print(f"Image: {w}x{h}\n")

# === Full vertical profile ===
print("=== Full vertical profile (every 20px, average across full width) ===")
for y in range(0, h, 20):
    pixels = [img.getpixel((x, y)) for x in range(0, w, 20)]
    r_avg = sum(p[0] for p in pixels) / len(pixels)
    g_avg = sum(p[1] for p in pixels) / len(pixels)
    b_avg = sum(p[2] for p in pixels) / len(pixels)
    brightness = (r_avg + g_avg + b_avg) / 3
    dark = sum(1 for p in pixels if sum(p)/3 < 20)
    light = sum(1 for p in pixels if sum(p)/3 > 100)
    total = len(pixels)
    print(f"  y={y:4d}: bright={brightness:5.1f}, dark%={dark*100//total:2d}%, light%={light*100//total:2d}%")

# === Bottom 200px horizontal detail ===
print("\n=== Bottom 200px: finding bright horizontal segments (>150px wide) ===")
for y in range(h - 200, h, 3):
    # Sample every 3px for speed, look for long bright stretches
    row_data = []
    for x in range(0, w, 3):
        px = img.getpixel((x, y))
        b = sum(px)/3
        row_data.append((x, b))
    
    # Find continuous bright segments
    segments = []
    start = None
    for x, b in row_data:
        if b > 60:
            if start is None:
                start = x
        else:
            if start is not None:
                seg_width = x - start
                if seg_width > 150:
                    segments.append((start, x, seg_width))
                start = None
    if start is not None:
        seg_width = w - start
        if seg_width > 150:
            segments.append((start, w, seg_width))
    
    if segments:
        print(f"  y={y}: {segments}")

# === Focus on the brightest areas ===
print("\n=== Detailed scan of right side (x>1200), bottom 200px ===")
for y in range(h - 200, h, 3):
    row_data = []
    for x in range(1200, w, 3):
        px = img.getpixel((x, y))
        b = sum(px)/3
        row_data.append((x, b))
    
    segments = []
    start = None
    for x, b in row_data:
        if b > 60:
            if start is None:
                start = x
        else:
            if start is not None:
                seg_width = x - start
                if seg_width > 50:
                    segments.append((start, x, seg_width))
                start = None
    if start is not None:
        seg_width = w - start
        if seg_width > 50:
            segments.append((start, w, seg_width))
    
    if segments:
        print(f"  y={y}: {segments}")

# === Check left side too (sidebar Claude Code) ===
print("\n=== Detailed scan of left side (x<600), bottom 200px ===")
for y in range(h - 200, h, 3):
    row_data = []
    for x in range(0, 600, 3):
        px = img.getpixel((x, y))
        b = sum(px)/3
        row_data.append((x, b))
    
    segments = []
    start = None
    for x, b in row_data:
        if b > 60:
            if start is None:
                start = x
        else:
            if start is not None:
                seg_width = x - start
                if seg_width > 50:
                    segments.append((start, x, seg_width))
                start = None
    if start is not None:
        seg_width = 600 - start
        if seg_width > 50:
            segments.append((start, 600, seg_width))
    
    if segments:
        print(f"  y={y}: left side segments: {segments}")

# === Precise analysis of the white rectangle at y~1070 ===
print("\n=== Detailed pixel analysis around y=1060-1080, x=480-720 ===")
for y in range(1062, h):
    colors = []
    for x in range(480, 720):
        px = img.getpixel((x, y))
        b = sum(px)/3
        colors.append(b)
    
    avg = sum(colors)/len(colors)
    
    # Find bright region
    bright = [(480+i, b) for i, b in enumerate(colors) if b > 150]
    if bright:
        left = min(p[0] for p in bright)
        right = max(p[0] for p in bright)
        print(f"  y={y}: area avg={avg:.0f}, bright pixels [{left}-{right}] width={right-left}, sample colors: {[img.getpixel((480+i*30, y)) for i in range(8)]}")
    else:
        if avg > 40:
            print(f"  y={y}: area avg={avg:.0f} (dimly lit)")

# === Figure out WHERE Claude Code panel is ===
# Look for panel borders (1px lines that separate panels)
print("\n=== Looking for panel borders (thin horizontal lines) ===")
for y in range(0, h, 1):
    row = [sum(img.getpixel((x, y))/3) for x in range(0, w, 2)]
    # Check for sharp transitions (border between panels)
    # A border would be a consistent color line
    avg = sum(row)/len(row)
    std = (sum((b - avg)**2 for b in row) / len(row))**0.5
    
    # Look for a row that's distinctly different from neighbors
    if y > 0 and y < h-1:
        prev_row = [sum(img.getpixel((x, y-1))/3) for x in range(0, w, 2)]
        next_row = [sum(img.getpixel((x, y+1))/3) for x in range(0, w, 2)]
        prev_avg = sum(prev_row)/len(prev_row)
        next_avg = sum(next_row)/len(next_row)
        
        # Check if this row is a thin line (border)
        if abs(avg - prev_avg) > 30 and abs(avg - next_avg) > 30:
            # This could be a panel border
            print(f"  y={y}: avg={avg:.1f}, prev={prev_avg:.1f}, next={next_avg:.1f}  <-- possible border")