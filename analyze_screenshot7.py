#!/usr/bin/env python3
"""Final targeted analysis."""

from PIL import Image
img = Image.open('/root/screenshot_vision.png')
w, h = img.size

# 1. Check the vertical profile at the blue icon area (x=440-460)
print("=== Blue icon area (x=440-460) vertical profile ===")
for y in range(0, h, 5):
    pixels = [sum(img.getpixel((x, y)))/3 for x in range(440, 460, 2)]
    avg_b = sum(pixels)/len(pixels)
    if avg_b > 10:
        sample = img.getpixel((450, y))
        print(f"  y={y}: avg={avg_b:.1f}, sample=RGB{sample}")

# 2. Check what's between the panel (y=540-680) and the white rect (y=1041-1070)
# Specifically, check if the dark area has ANY content
print("\n=== Dark area (y=680-950) brightness peaks ===")
for y in range(680, 950):
    max_b = 0
    for x in range(0, w, 5):
        b = sum(img.getpixel((x, y)))/3
        max_b = max(max_b, b)
    if max_b > 20:
        # Find where
        for x in range(0, w, 3):
            b = sum(img.getpixel((x, y)))/3
            if b > 20:
                print(f"  y={y}: max_b={max_b:.0f}, at x={x}, RGB={img.getpixel((x, y))}")
                break

# 3. Check the white rect area - is it INSIDE a taskbar?
print("\n=== Checking if white rect is inside a larger panel ===")
# Check x=440-460 (blue icon area) - does it extend upward
print("Blue icon vertical extent (x=450):")
for y in range(1035, 1080):
    px = img.getpixel((450, y))
    if sum(px)/3 > 30:
        print(f"  y={y}: RGB={px}")

# Check x=550 (inside the white rect)
print("\nWhite rectangle vertical extent (x=550):")
for y in range(1030, 1080):
    px = img.getpixel((550, y))
    if sum(px)/3 > 50:
        print(f"  y={y}: RGB={px}")

# 4. Check the purple pixels at 540 more carefully - are they part of a Claude panel?
print("\n=== Purple panel area detail ===")
# These are the Claude Code brand purple pixels
# In VS Code, Claude Code uses a purple header/border
for y in range(535, 600):
    purple_count = 0
    for x in range(80, 400, 3):
        r, g, b = img.getpixel((x, y))
        if b > 180 and r < 150 and g < 150 and abs(r-g) < 50:
            purple_count += 1
    if purple_count > 0:
        print(f"  y={y}: {purple_count} purple pixels")

# 5. Check if x=440-460 blue is an icon in the taskbar
print("\n=== Icon at x=440-460, y=1042-1070 ===")
# This blue icon could be a search icon, Claude icon, or taskbar app
# Check if it extends up into the taskbar area
for y in range(940, 1045):
    for x in range(440, 460, 2):
        px = img.getpixel((x, y))
        if sum(px)/3 > 50:
            print(f"  ({x}, {y}): RGB={px}")
            break

# 6. Check what's at x=440-460 ABOVE the taskbar
print("\n=== Icon area tracing upward from y=1040 ===")
for y in range(1040, 900, -5):
    px = img.getpixel((450, y))
    print(f"  y={y}: RGB={px}, brightness={sum(px)/3:.0f}")

# 7. Check the area around the white rectangle for taskbar context
print("\n=== Taskbar-like elements around white rectangle ===")
# Windows taskbar typically has:
# - Start button (bottom-left)
# - Search box
# - Task view button
# - System tray (bottom-right with clock)
# Check bottom-left for Start button
for x in range(0, 100, 3):
    for y in range(1040, 1080):
        px = img.getpixel((x, y))
        if sum(px)/3 > 50:
            print(f"  Start area ({x}, {y}): RGB={px}")
            break
    else:
        continue
    break

# Check bottom-right for clock
print("\nClock area (x=1860-1920, y=1060-1080):")
for y in range(1040, 1080, 2):
    for x in range(1860, 1920, 3):
        px = img.getpixel((x, y))
        if sum(px)/3 > 100:
            print(f"  ({x}, {y}): RGB={px}")
            break
    else:
        continue
    break

# 8. The MOST important check: is the white rect Claude's chat input?
# Claude Code uses a distinctive color scheme: purple headers, white input
# In VS Code's bottom panel, the chat input would span the panel width
# If it's in a narrow panel, it would be narrow
# Check if there's something ABOVE the white rect that looks like chat messages
print("\n=== Chat messages above white rectangle? ===")
# Check for text-like patterns at y=1000-1040 above the white rect
for y in range(1000, 1040):
    text_pixels = []
    for x in range(480, 695, 2):
        px = img.getpixel((x, y))
        b = sum(px)/3
        if b > 100 and b < 250:
            text_pixels.append((x, b))
    if len(text_pixels) > 5:
        print(f"  y={y}: {len(text_pixels)} mid-bright pixels above white rect (potential text)")

# 9. Look for Claude Code header in the panel area
print("\n=== Checking for Claude tab/header in panel ===")
# In the bottom panel, tabs are usually at the top (y~540-565)
# Check for the purple accent that Claude uses
for y in range(540, 580):
    for x in range(80, 400, 2):
        px = img.getpixel((x, y))
        r, g, b = px
        # Claude's purple accent line
        if r > 80 and r < 120 and g > 80 and g < 120 and b > 190 and b < 260:
            print(f"  Claude purple pixel at ({x}, {y}): RGB=({r},{g},{b})")
            break

# 10. Check the top of the bright band at y=950
print("\n=== y=950 bright band analysis ===")
for y in [948, 949, 950, 951, 952, 953, 954, 955]:
    px = img.getpixel((960, y))
    print(f"  y={y}: RGB={px}")

# 11. FINAL CHECK: Could the panel be BELOW the bright band at y=950?
print("\n=== Panel below y=950 check ===")
# If the bright band at y=950 is a panel header/divider,
# then the panel content could be at y=950-1080
for y in range(950, 1080, 5):
    brights = [sum(img.getpixel((x, y)))/3 for x in range(100, 700, 10)]
    avg = sum(brights) / len(brights)
    print(f"  y={y}: avg_bright(x=100-700)={avg:.1f}")

# 12. Check around x=0-80 for the VS Code side bar / activity bar
print("\n=== VS Code activity bar (x=30-70) bottom portion ===")
for y in range(500, 700, 5):
    brights = [sum(img.getpixel((x, y)))/3 for x in range(30, 70, 3)]
    avg = sum(brights) / len(brights)
    if avg > 5:
        print(f"  y={y}: avg={avg:.1f}")