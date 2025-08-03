"""
Default configurations for ControlNet
Used by setup command to create initial config files
"""


def create_default_controlnet_config() -> dict:
    """
    Create default ControlNet configuration

    This function is called by the setup command to create initial configs.
    After creation, all values MUST come from the config file.
    These are NOT hardcoded defaults - they're initial config values.

    Returns:
        dict: Initial configuration to be saved as YAML
    """
    return {
        "# ControlNet Configuration": None,
        "# All values have sensible defaults but can be customized": None,

        # Default parameter values for all ControlNet operations
        "defaults": {
            "negative_prompt": "",
            "controlnet_strength": 1.0,
            "strength": 0.8,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
        },

        # Extractor settings
        "extractors": {
            "canny": {
                # Threshold values with sensible defaults
                "low_threshold_min": 0,
                "low_threshold_max": 255,
                "high_threshold_min": 0,
                "high_threshold_max": 255,
                # Default Canny thresholds
                "default_low_threshold": 100,
                "default_high_threshold": 200,
                # Kernel parameters for dilation
                "kernel_size": 3,
                "dilation_iterations": 1,
            },

            "blur": {
                # Blur types
                "valid_types": ["gaussian", "box", "motion"],
                # Motion blur parameters with defaults
                "motion_kernel_multiplier": 2,
                "motion_kernel_offset": 1,
                # Default blur radius
                "default_radius": 5,
            },

            "depth": {
                # Model for depth extraction
                "model_id": "Intel/dpt-large",
                # Depth normalization
                "normalize": True,
                "invert": False,
            },

            "resampling": {
                # PIL resampling method
                "method": "LANCZOS",
            },
        },

        # Pipeline settings
        "pipelines": {
            # SDXL dimension requirements
            "dimension_multiple": 8,
            # FAIL LOUD on invalid dimensions - NO AUTO-RESIZE
            "validate_dimensions": True,
        },

        # Model references - these are defaults that can be customized
        # Users can change these to use different ControlNet models
        # Example: "canny": "your-username/your-custom-canny-model"
        "models": {
            "sdxl": {
                # Default HuggingFace model IDs - CUSTOMIZABLE
                "canny": "diffusers/controlnet-canny-sdxl-1.0",
                "depth": "diffusers/controlnet-depth-sdxl-1.0",
                "openpose": "diffusers/controlnet-openpose-sdxl-1.0",
                # Add custom models here:
                # "custom_type": "your-model-id",
            },
            # Add support for other model types:
            # "sd15": {
            #     "canny": "lllyasviel/sd-controlnet-canny",
            #     ...
            # }
        },

        # Strategy settings
        "strategy": {
            # Default values for strategies - users can override
            "default_extract_at_each_step": True,
        },

        # VRAM overhead estimates (MB)
        "vram_overhead": {
            "model_load": 2000,  # Per ControlNet model
            "operation_active": 1500,  # Additional for active operations
        },

        # Calculation constants
        "calculations": {
            "megapixel_divisor": 1000000,  # 1e6 for MP calculations
        },
    }


def update_vram_strategies_with_defaults() -> dict:
    """
    Create default operation_estimates section for vram_strategies.yaml

    This is added to existing vram_strategies.yaml if missing
    """
    return {
        # VRAM operation estimates (MB)
        "operation_estimates": {
            # Base VRAM usage for operations by model type
            "sdxl": {
                "generate": 6000,
                "inpaint": 5500,
                "img2img": 5000,
                "refine": 4000,
                "enhance": 3500,
                "controlnet_generate": 8500,
                "controlnet_inpaint": 8000,
                "controlnet_img2img": 7500,
            },

            "sd3": {
                "generate": 8000,
                "inpaint": 7500,
                "img2img": 7000,
                "refine": 5000,
                "enhance": 4500,
                "controlnet_generate": 10500,
                "controlnet_inpaint": 10000,
                "controlnet_img2img": 9500,
            },

            "flux": {
                "generate": 12000,
                "inpaint": 11000,
                "img2img": 10000,
                "refine": 8000,
                "enhance": 7000,
                "controlnet_generate": 14500,
                "controlnet_inpaint": 14000,
                "controlnet_img2img": 13500,
            },

            "sd15": {
                "generate": 3000,
                "inpaint": 2800,
                "img2img": 2500,
                "refine": 2000,
                "enhance": 1800,
                "controlnet_generate": 4500,
                "controlnet_inpaint": 4300,
                "controlnet_img2img": 4000,
            },

            "sd2": {
                "generate": 4000,
                "inpaint": 3800,
                "img2img": 3500,
                "refine": 2500,
                "enhance": 2300,
                "controlnet_generate": 5500,
                "controlnet_inpaint": 5300,
                "controlnet_img2img": 5000,
            },

            # Resolution scaling factors (multiplier per megapixel)
            "resolution_scaling": {
                "per_megapixel": 150,  # Additional MB per megapixel
                "batch_size_multiplier": 0.8,  # Additional factor per batch item
            },

            # LoRA overhead
            "lora_overhead": 200,  # MB per LoRA
        }
    }
