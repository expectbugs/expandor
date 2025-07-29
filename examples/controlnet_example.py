"""
Example of using ControlNet with Expandor

This example shows how to use ControlNet for structure-preserving expansion.
ALL parameters come from configuration files - NO HARDCODED VALUES.
FAIL LOUD: Any missing configuration will cause explicit errors.
"""

from pathlib import Path
from PIL import Image, ImageDraw

from expandor import Expandor, ExpandorConfig
from expandor.adapters import DiffusersPipelineAdapter
from expandor.processors.controlnet_extractors import ControlNetExtractor
from expandor.utils.config_loader import ConfigLoader


def create_test_image():
    """Create a simple test image"""
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    
    # Draw a simple house
    # House body
    draw.rectangle([150, 250, 350, 400], fill="lightblue", outline="black", width=2)
    # Roof  
    draw.polygon([(150, 250), (250, 150), (350, 250)], fill="red", outline="black", width=2)
    # Door
    draw.rectangle([225, 320, 275, 400], fill="brown", outline="black")
    # Windows
    draw.rectangle([180, 280, 220, 320], fill="yellow", outline="black")
    draw.rectangle([280, 280, 320, 320], fill="yellow", outline="black")
    
    return img


def main():
    # Get config directory path
    config_dir = Path(__file__).parent.parent / "config"
    
    # Load configuration
    config_loader = ConfigLoader(config_dir)
    try:
        controlnet_config = config_loader.load_config_file("controlnet_config.yaml")
        quality_config = config_loader.load_quality_preset("high")
    except FileNotFoundError as e:
        print(f"Configuration error: {e}")
        print("Run 'expandor --setup' to create default configuration")
        return
    
    # Create or load your image
    print("Creating test image...")
    image = create_test_image()
    image.save("test_house.png")
    
    # Initialize adapter with SDXL model
    print("Initializing adapter...")
    adapter = DiffusersPipelineAdapter(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype="float16",
        device="cuda"
    )
    
    # Load ControlNet model from config
    print("Loading ControlNet model...")
    try:
        if "models" not in controlnet_config:
            raise ValueError("models section not found in controlnet_config.yaml")
        if "sdxl" not in controlnet_config["models"]:
            raise ValueError("sdxl section not found in models")
        model_refs = controlnet_config["models"]["sdxl"]
        
        if "canny" not in model_refs:
            raise ValueError("canny model not found in sdxl models config")
        canny_model = model_refs["canny"]
        adapter.load_controlnet(canny_model, controlnet_type="canny")
    except Exception as e:
        print(f"Failed to load ControlNet: {e}")
        print("Make sure you have installed: pip install 'diffusers[controlnet]'")
        return
    
    # Extract control signal (uses config defaults if not specified)
    print("Extracting edges...")
    try:
        extractor = ControlNetExtractor()
        
        # Extract edges - parameters are optional, defaults from config
        edges = extractor.extract_canny(
            image, 
            low_threshold=100,  # Optional - defaults from config if not specified
            high_threshold=200,  # Optional - defaults from config if not specified
            dilate=False,  # Optional - default is True
            l2_gradient=False  # Optional - default is False
        )
        edges.save("test_house_edges.png")
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        return
    
    # Configure expansion with ControlNet
    config = ExpandorConfig(
        source_image=image,
        target_resolution=(1024, 768),  # Wider aspect ratio
        strategy="controlnet_progressive",
        prompt="a beautiful detailed house with garden, high quality, 4k",
        negative_prompt="blurry, low quality, distorted",  # Optional - defaults to ""
        quality_preset="high",
        num_inference_steps=quality_config["generation"]["num_inference_steps"],
        guidance_scale=quality_config["generation"]["guidance_scale"],
        strategy_params={
            "controlnet_config": {
                "control_type": "canny",  # Optional - defaults to "canny"
                "controlnet_strength": quality_config["controlnet"]["controlnet_strength"],  # Optional - defaults from config
                "extract_at_each_step": True,  # Optional - defaults from config
                # Canny parameters are optional - will use defaults if not specified
                "canny_low_threshold": 100,
                "canny_high_threshold": 200,
                "canny_dilate": False,
                "canny_l2_gradient": False
            },
            "strength": quality_config["expansion"]["denoising_strength"],
            "steps": quality_config["generation"]["num_inference_steps"]
        }
    )
    
    # Perform expansion
    print("Expanding image with ControlNet guidance...")
    expandor = Expandor(adapter)
    
    try:
        result = expandor.expand(config)
        result.save("test_house_expanded.png")
        print("Success! Saved expanded image to test_house_expanded.png")
        
    except Exception as e:
        print(f"Expansion failed: {e}")
        
        # If it's a dimension issue, show proper error
        if "dimension_multiple" in str(e):
            print("\nDimensions must be multiples of the configured value.")
            print("Check controlnet_config.yaml for the required multiple.")


if __name__ == "__main__":
    main()