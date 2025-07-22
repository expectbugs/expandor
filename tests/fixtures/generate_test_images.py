"""
Generate test images for Expandor testing
"""

from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

def generate_test_images():
    """Generate various test images"""
    fixtures_dir = Path(__file__).parent
    fixtures_dir.mkdir(exist_ok=True)
    
    # 1. Simple gradient image (1024x1024)
    gradient = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for i in range(1024):
        gradient[i, :, 0] = int(i * 255 / 1024)  # Red gradient vertical
        gradient[:, i, 1] = int(i * 255 / 1024)  # Green gradient horizontal
    gradient_img = Image.fromarray(gradient)
    gradient_img.save(fixtures_dir / "gradient_1024x1024.png")
    
    # 2. SDXL-like image (1344x768)
    sdxl_img = Image.new('RGB', (1344, 768), color='skyblue')
    draw = ImageDraw.Draw(sdxl_img)
    # Add some features
    draw.ellipse([300, 200, 1044, 568], fill='green')  # "Landscape"
    draw.rectangle([500, 400, 844, 768], fill='brown')  # "Ground"
    sdxl_img.save(fixtures_dir / "landscape_1344x768.png")
    
    # 3. Portrait image (768x1344)
    portrait = Image.new('RGB', (768, 1344), color='lightgray')
    draw = ImageDraw.Draw(portrait)
    draw.ellipse([184, 200, 584, 600], fill='peachpuff')  # "Face"
    portrait.save(fixtures_dir / "portrait_768x1344.png")
    
    # 4. Small test image (512x512)
    small = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(small)
    for i in range(0, 512, 64):
        color = 'black' if (i // 64) % 2 == 0 else 'white'
        draw.rectangle([i, 0, i+64, 512], fill=color)
    small.save(fixtures_dir / "checkerboard_512x512.png")
    
    print(f"Generated test images in {fixtures_dir}")

if __name__ == "__main__":
    generate_test_images()