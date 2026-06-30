#!/usr/bin/env python3
"""Analyze VS Code screenshot to find Claude Code chat input field."""

from PIL import Image
import sys

def analyze_screenshot(path):
    img = Image.open(path)
    width, height = img.size
    print(f"Image size: {width}x{height}")
    
    # Get the bottom 300 pixels - likely panel area
    # VS Code panel typically at bottom with terminal/chat
    panel_region = img.crop((0, height - 400, width, height))
    
    # Let's analyze in sections to find the chat input
    # We'll look for:
    # 1. Dark background (VS Code dark theme)
    # 2. Text input areas (lighter rectangles)
    # 3. Send button or "Type a message" placeholder
    
    print("\n=== Bottom 400px pixel analysis ===")
    print(f"Panel region: (0, {height-400}, {width}, {height})")
    
    # Sample horizontal strips to understand layout
    strips = 20
    strip_h = 400 // strips
    for i in range(strips):
        y = height - 400 + i * strip_h
        center_x = width // 2
        # Get pixels at center of this strip
        pixels = []
        for x_offset in range(0, width, 10):
            px = img.getpixel((x_offset, y))
            pixels.append(px)
        
        # Find the most common color in this strip
        from collections import Counter
        color_counts = Counter(pixels)
        top_colors = color_counts.most_common(3)
        
        # Detect if there's a lighter input field
        lighter_spots = []
        for x_offset in range(0, width, 5):
            px = img.getpixel((x_offset, y))
            brightness = sum(px) / 3
            if brightness > 80:  # lighter than dark bg
                lighter_spots.append((x_offset, brightness))
        
        y_pos = y
        if lighter_spots:
            avg_bright = sum(b for _, b in lighter_spots) / len(lighter_spots)
            print(f"  y={y_pos}: top colors={top_colors}, lighter_spots={len(lighter_spots)} spots, avg_bright={avg_bright:.1f}")
        else:
            print(f"  y={y_pos}: top colors={top_colors}, all dark (no lighter spots)")
    
    # Now let's do a detailed scan for the input field
    # The input field in Claude Code chat is typically a textarea at the bottom
    # It's usually a lighter rectangle on a dark background
    
    print("\n=== Detailed horizontal scan at various y positions ===")
    for y_scan in range(height - 100, height - 5, 5):
        row_brights = []
        for x in range(0, width, 3):
            r, g, b = img.getpixel((x, y_scan))
            bright = (r + g + b) / 3
            row_brights.append((x, bright))
        
        avg_bright = sum(b for _, b in row_brights) / len(row_brights)
        
        # Find clusters of bright pixels (potential input field)
        bright_threshold = 60
        in_bright = False
        bright_segments = []
        seg_start = 0
        for x, b in row_brights:
            if b > bright_threshold and not in_bright:
                seg_start = x
                in_bright = True
            elif b <= bright_threshold and in_bright:
                bright_segments.append((seg_start, x))
                in_bright = False
        if in_bright:
            bright_segments.append((seg_start, width))
        
        # Filter to significant segments (>50px wide)
        significant = [(s, e) for s, e in bright_segments if e - s > 50]
        
        if significant:
            print(f"  y={y_scan}: avg_bright={avg_bright:.1f}, segments: {significant}")
    
    # Let's also check if there's a distinctive "Claude Code" header or icon
    # Look for purple/blue colors (Claude brand color)
    print("\n=== Looking for Claude purple/blue accent colors ===")
    for y_check in range(0, height, 20):
        for x_check in range(0, width, 20):
            r, g, b = img.getpixel((x_check, y_check))
            # Claude purple-ish: #6C5CE7 or similar
            # Also looking for VS Code blue accents
            if (r > 100 and g > 80 and b > 150 and r < 150):  # purplish
                print(f"  Purple pixel at ({x_check}, {y_check}): RGB=({r},{g},{b})")
            if (r < 80 and g > 100 and b > 180):  # blue accent
                print(f"  Blue pixel at ({x_check}, {y_check}): RGB=({r},{g},{b})")
    
    # Comprehensive bottom row analysis
    print("\n=== Last 10 rows pixel by pixel (every 20px) ===")
    for y_last in range(height - 10, height):
        row = []
        for x in range(0, width, 20):
            px = img.getpixel((x, y_last))
            row.append(f"({px[0]},{px[1]},{px[2]})")
        print(f"  y={y_last}: {' '.join(row)}")
    
    # Analyze the right side specifically - Claude Code often as sidebar or bottom-right
    # Let's check the right 1/3 of the screen
    right_start = width * 2 // 3
    print(f"\n=== Right side ({right_start}-{width}) bottom 200px ===")
    for y_rs in range(height - 200, height, 10):
        row_pixels = []
        for x_rs in range(right_start, width, 15):
            px = img.getpixel((x_rs, y_rs))
            row_pixels.append(sum(px) / 3)
        avg = sum(row_pixels) / len(row_pixels)
        # find lighter rectangle (input field)
        bright_regions = []
        for i, b in enumerate(row_pixels):
            if b > 60:
                bright_regions.append(right_start + i * 15)
        if bright_regions and len(bright_regions) > 10:
            print(f"  y={y_rs}: avg={avg:.1f}, bright region: [{bright_regions[0]}-{bright_regions[-1]}] ({len(bright_regions)} px)")

if __name__ == '__main__':
    analyze_screenshot('/root/screenshot_vision.png')