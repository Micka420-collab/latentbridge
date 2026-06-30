#!/usr/bin/env python3
"""CRITICAL: Find the chat input - check bottom of Claude panel."""

from PIL import Image
img = Image.open('/root/screenshot_vision.png')
w, h = img.size

# 1. Check the full panel area y=540-690 for ANY input-like field
print("=== Panel bottom (y=620-690) DETAIL ===")
for y in range(620, 690):
    for x in range(0, w, 2):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if b > 100:  # Medium brightness
            print(f"  y={y}: x={x}, RGB={px}, bright={b:.0f}")
            break

# 2. Check for the VS Code status bar (usually at the bottom of the window)
print("\n=== VS Code Status Bar check ===")
# Status bar is typically a thin colored strip at window bottom
for y in range(680, 720):
    bright_pixels = []
    for x in range(0, w, 5):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if b > 50:
            bright_pixels.append((x, b, px))
    if len(bright_pixels) > 5:
        print(f"  y={y}: {len(bright_pixels)} bright pixels, first: {bright_pixels[0]}")

# 3. Check around y=660-670 for an input field specifically
print("\n=== Input field search in panel bottom (y=650-685) ===")
# Look for a horizontal strip that's evenly lit (like a text input)
for y in range(650, 685):
    row = [sum(img.getpixel((x, y)))/3 for x in range(0, w, 3)]
    # Check for a uniform bright region
    for i in range(0, len(row)):
        if row[i] > 80 and i < len(row)-20:
            segment = row[i:i+20]
            if all(50 < v < 200 for v in segment):
                x_start = i * 3
                print(f"  y={y}: x≈{x_start}, avg_bright={sum(segment)/len(segment):.0f}")
                # Check colors in this segment
                px = img.getpixel((x_start+10, y))
                print(f"    sample RGB={px}")
                break

# 4. Check if the Claude panel extends BELOW y=690
print("\n=== Claude activity bar icon position (x=48) ===")
# The orange at x=48 - let me trace it to find the active panel
for y in range(680, 960):
    px = img.getpixel((48, y))
    if px[0] > 200 and px[1] < 150 and px[2] < 50:  # Orange-red
        print(f"  y={y}: RGB={px}")

# 5. Check the x=100-300 range in the panel area for Claude Code content
print("\n=== Claude panel content (x=100-300, y=560-680) ===")
for y in range(560, 680, 2):
    row = [sum(img.getpixel((x, y)))/3 for x in range(100, 300, 2)]
    avg = sum(row)/len(row)
    if avg > 10:
        print(f"  y={y}: avg={avg:.0f}")

# 6. Check for a dark-themed input field in the panel
print("\n=== Dark input field search in panel ===")
# In dark theme, the input field would be slightly lighter than the background
# Check for subtle brightness changes in the bottom of the panel
for y in range(660, 690):
    for x in range(100, 600, 5):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if 30 < b < 100:  # Medium-dark (potential dark input field)
            print(f"  y={y}: x={x}, RGB={px}")
            break

# 7. Check what's at y=1009-1010 (had "chat messages" above the white rect)
print("\n=== Content at y=1009-1010 (chat messages above white rect) ===")
for y in [1008, 1009, 1010, 1011]:
    brights = [(x, sum(img.getpixel((x, y)))/3, img.getpixel((x, y))) for x in range(100, 700, 3)]
    brights = [(x, b, px) for x, b, px in brights if b > 50]
    if brights:
        print(f"  y={y}: {len(brights)} bright pixels")
        for x, b, px in brights[:5]:
            print(f"    x={x}: bright={b:.0f}, RGB={px}")

# 8. Check if the bright band at y=950 is actually the WINDOWS taskbar
print("\n=== Windows Taskbar verification ===")
# Windows taskbar typically has a specific look
# Check for the clock digits in the bottom-right corner
for x in range(1840, 1920, 2):
    for y in range(1040, 1080, 2):
        px = img.getpixel((x, y))
        r, g, b = px
        # Clock digits would be white or bright on a dark background
        if r > 200 and g > 200 and b > 200:
            print(f"  Clock pixel at ({x}, {y}): RGB={px}")

# Also check for the taskbar system tray icons
print("\n=== System tray icons (x=1800-1920, y=1040-1065) ===")
for x in range(1800, 1920, 4):
    for y in range(1040, 1065, 3):
        px = img.getpixel((x, y))
        r, g, b = px
        if max(r, g, b) > 200:
            print(f"  ({x}, {y}): RGB={px}")
            break

# 9. Check white rect left border area - what's right next to it
print("\n=== White rectangle left neighbor check at x=450-480 ===")
for y in range(1040, 1072):
    px = img.getpixel((470, y))
    print(f"  y={y}: ({470}, {y}): RGB={px}")