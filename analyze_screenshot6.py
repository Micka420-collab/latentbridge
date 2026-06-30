#!/usr/bin/env python3
"""Check bottom panel area y=540-680 for Claude Code chat input."""

from PIL import Image
img = Image.open('/root/screenshot_vision.png')
w, h = img.size

# Check the panel area y=540-680 for any input-like rectangles
print("=== Bottom panel (y=540-680) horizontal segments ===")
for y in range(540, 680, 2):
    segments = []
    in_light = False
    start = 0
    for x in range(0, w, 1):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if b > 180:  # Light pixel
            if not in_light:
                start = x
                in_light = True
        else:
            if in_light:
                width = x - start
                if width > 80:  # Wider than 80px
                    segments.append((start, x, width))
                in_light = False
    if in_light:
        width = w - start
        if width > 80:
            segments.append((start, w, width))
    
    if segments:
        print(f"  y={y}: {segments}")

# Check between y=900-1080 for ANY input-like fields
print("\n=== All light rectangles in y=900-1080 ===")
for y in range(900, h, 2):
    segments = []
    in_light = False
    start = 0
    for x in range(0, w, 1):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if b > 180:
            if not in_light:
                start = x
                in_light = True
        else:
            if in_light:
                width = x - start
                if width > 50:
                    segments.append((start, x, width))
                in_light = False
    if in_light:
        width = w - start
        if width > 50:
            segments.append((start, w, width))
    
    if segments:
        for s, e, width in segments:
            if width > 50:
                print(f"  y={y}: x=[{s}-{e}] width={width}")

# Now check right around the panel bottom
print("\n=== Detailed look at bottom of panel (y=650-680) ===")
for y in range(650, 680):
    segments = []
    in_light = False
    start = 0
    for x in range(0, w, 1):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if b > 50:  # Lower threshold for this darker area
            if not in_light:
                start = x
                in_light = True
        else:
            if in_light:
                width = x - start
                if width > 30:
                    segments.append((start, x, width))
                in_light = False
    if in_light:
        width = w - start
        if width > 30:
            segments.append((start, w, width))
    
    if segments:
        print(f"  y={y}: segments={segments}")
        for s, e, width in segments[:3]:
            px = img.getpixel((s+5, y))
            print(f"    [{s}-{e}]: sample color = RGB{px}")

# Check for the Claude purple icon/header in the panel 
print("\n=== Detailed purple region in panel area ===")
# We found purple at y=540-596. Let's see what it looks like
for y in range(538, 600, 2):
    for x in range(80, 400, 2):
        px = img.getpixel((x, y))
        r, g, b = px
        if b > 180 and r < 150 and g < 150:
            print(f"  ({x}, {y}): RGB=({r},{g},{b})")
            break

# Check the taskbar-like area more carefully
print("\n=== Color analysis of bright band at y=950-985 ===")
for y in [948, 949, 950, 951, 952, 953, 954, 960, 982, 983, 984, 985, 986]:
    avg = sum(sum(img.getpixel((x, y))/3) for x in range(0, w, 20)) / (w//20)
    sample = img.getpixel((960, y))
    print(f"  y={y}: avg={avg:.1f}, center pixel={sample}")

# FINAL: The comprehensive picture
print("\n\n=== COMPREHENSIVE LAYOUT MAP ===")
# Based on all data, map out what we see

# y=0-470: VS Code window (title bar + tabs + editor in light theme)
# y=470-540: Panel border/tabs (dark)
# y=540-680: Bottom panel (mixed dark and bright - this is where Claude Code might be)
# y=680-950: Very dark area (desktop background? or VS Code status bar?)
# y=950-985: Bright bands (Windows taskbar)
# y=985-1040: Dark area
# y=1041-1070: White rectangle (Windows search box in taskbar, or Claude Code input)
# y=1070-1080: Dark bottom edge

# The white rectangle at x=480-695, y=1041-1070 is the ONLY prominent input-like UI element
# Given it's at the very bottom, it's either:
# 1. A Windows taskbar search box
# 2. A chat input field

# The Claude purple pixels at y=540-596 in the panel area suggest Claude IS active
# But the chat input would be at the bottom of the panel

print()
print("The white rectangle at:")
print(f"  x=[480, 695], y=[1041, 1070]")
print(f"  width=215px, height=30px")
print(f"  Color: RGB~(243,243,243)")
print(f"  Left border: ~x=470 (dark)")
print(f"  Right border: ~x=700 (dark)")
print(f"  Blue icon left of it at x=440-465")
print()
print("Center point:")
print(f"  x={(480+695)//2}, y={(1041+1070)//2}")
print(f"  = (587, 1056)")