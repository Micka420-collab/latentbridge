#!/usr/bin/env python3
"""Slim, fast analysis - get the answer now."""

from PIL import Image
img = Image.open('/root/screenshot_vision.png')
w, h = img.size

print(f"Image: {w}x{h}\n")

# 1. Check the right side for a side panel
print("=== Right side scan (x=1500-1920) ===")
for y in range(0, h, 20):
    brightnesses = [sum(img.getpixel((x, y)))/3 for x in range(1500, 1920, 10)]
    avg = sum(brightnesses) / len(brightnesses)
    print(f"  y={y:4d}: avg={avg:.1f}")

# 2. Check x=1200-1500 (potential intermediate panel)
print("\n=== Middle-right side scan (x=1200-1500) ===")
for y in range(0, h, 20):
    brightnesses = [sum(img.getpixel((x, y)))/3 for x in range(1200, 1500, 10)]
    avg = sum(brightnesses) / len(brightnesses)
    print(f"  y={y:4d}: avg={avg:.1f}")

# 3. Check around x=0-70 (activity bar)
print("\n=== Activity bar (x=0-70) ===")
for y in range(0, h, 20):
    brightnesses = [sum(img.getpixel((x, y)))/3 for x in range(0, 70, 5)]
    avg = sum(brightnesses) / len(brightnesses)
    print(f"  y={y:4d}: avg={avg:.1f}")

# 4. Check x=70-300 (sidebar area)
print("\n=== Left sidebar (x=70-300) ===")
for y in range(0, h, 20):
    brightnesses = [sum(img.getpixel((x, y)))/3 for x in range(70, 300, 10)]
    avg = sum(brightnesses) / len(brightnesses)
    print(f"  y={y:4d}: avg={avg:.1f}")

# 5. Check a few specific coordinates for color palette
print("\n=== Color samples at key positions ===")
positions = [
    # Title bar area
    (960, 10), (960, 30), (960, 50), (960, 100),
    # Tab area
    (960, 130), (960, 160), (960, 190),
    # Editor area
    (960, 220), (960, 300), (960, 400),
    # Panel area
    (960, 480), (960, 500), (960, 520), (960, 540), (960, 560), (960, 580), (960, 600), (960, 640), (960, 660),
    # Below panel
    (960, 700), (960, 750), (960, 800), (960, 850), (960, 900), (960, 950),
    # At the white rectangle
    (588, 1050), (588, 1055), (588, 1060), (588, 1065),
    # Around it
    (480, 1055), (700, 1055), (588, 1040),
    # Very bottom
    (960, 1070), (960, 1075), (960, 1079),
    # Bottom-right (clock area)
    (1850, 1070),
    # Check for Claude panel in right sidebar
    (1700, 1050),
]
for x, y in positions:
    px = img.getpixel((x, y))
    print(f"  ({x:4d}, {y:4d}): RGB={px} (brightness={sum(px)/3:.0f})")

# 6. Find the exact center of the white rectangle and check what's around it
print("\n=== White rectangle detailed analysis ===")
# Horizontal profile at y=1055
print("Horizontal profile at y=1055:")
for x in range(440, 740, 5):
    px = img.getpixel((x, 1055))
    print(f"  x={x}: RGB={px}")

# Vertical profile at x=588 (center)
print("\nVertical profile at x=588:")
for y in range(1030, h):
    px = img.getpixel((588, y))
    print(f"  y={y}: RGB={px}")

# 7. Check what's at the very bottom-right corner (Windows clock area)
print("\n=== Taskbar area check ===")
# If there's a Windows taskbar, the bottom-right should have clock/notification icons
for y in range(1040, h):
    for x in range(1820, w, 2):
        px = img.getpixel((x, y))
        if sum(px)/3 > 50:
            print(f"  ({x}, {y}): RGB={px}")
            break

# 8. Look for the Claude Code panel header or Claude icon
print("\n=== Searching for Claude icon/brand ===")
# Claude's icon is a stylized 'C' or claude color (#6C5CE7 purple)
# Look for purple/violet pixels in the panel area
for y in range(540, 680, 2):
    for x in range(0, w, 2):
        r, g, b = img.getpixel((x, y))
        if b > 180 and r < 150 and g < 150 and abs(r-g) < 30:
            print(f"  Purple pixel at ({x}, {y}): RGB=({r},{g},{b})")

# 9. Check the second bright spot found earlier at x=1840, y=1060-1066
print("\n=== Right-side bright spot at x~1840, y~1060 ===")
for y in range(1055, 1075):
    for x in range(1830, 1910):
        px = img.getpixel((x, y))
        if sum(px)/3 > 100:
            print(f"  ({x}, {y}): RGB={px}")