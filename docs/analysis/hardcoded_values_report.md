# Hardcoded Values Scan Report

Total findings: 829
Files scanned: 68
Files with issues: 53

## Summary by Type
- ast_constant: 416
- direct_assignment: 37
- field_default: 119
- function_default: 38
- get_default: 32
- get_with_default: 81
- if_comparison: 9
- math_operations: 93
- or_pattern: 1
- range_calls: 3

## Detailed Findings

### adapters/a1111_adapter.py
Found 2 issues:

- Line 42: get_with_default: logging.getLogger(__name__
  ```python
  logger = kwargs.get("logger", logging.getLogger(__name__))
  ```
- Line 290: get_with_default: config_manager.get_value("adapters.a1111.vram_estimates.default"
  ```python
  return estimates.get("default", config_manager.get_value("adapters.a1111.vram_estimates.default"))
  ```

### adapters/base_adapter.py
Found 16 issues:

- Line 111: function_default: 7.5
  ```python
  def controlnet_inpaint(
  ```
- Line 118: field_default: controlnet_conditioning_scale
  ```python
  controlnet_conditioning_scale: float = 1.0,
  ```
- Line 119: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 119: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 120: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 120: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 121: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 121: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 149: function_default: 7.5
  ```python
  def controlnet_img2img(
  ```
- Line 155: field_default: controlnet_conditioning_scale
  ```python
  controlnet_conditioning_scale: float = 1.0,
  ```
- Line 156: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 156: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 157: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 157: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 158: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 158: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```

### adapters/comfyui_adapter.py
Found 1 issues:

- Line 42: get_with_default: logging.getLogger(__name__
  ```python
  logger = kwargs.get("logger", logging.getLogger(__name__))
  ```

### adapters/diffusers_adapter.py
Found 43 issues:

- Line 68: direct_assignment: guidance_scale
  ```python
  guidance_scale=7.5  # REQUIRED
  ```
- Line 269: Numeric constant: 1024
  ```python
  "optimal_resolution": (1024, 1024),
  ```
- Line 269: Numeric constant: 1024
  ```python
  "optimal_resolution": (1024, 1024),
  ```
- Line 274: math_operations: 3
  ```python
  "patterns": ["stable-diffusion-3", "sd3", "sd-3"],
  ```
- Line 274: math_operations: 3
  ```python
  "patterns": ["stable-diffusion-3", "sd3", "sd-3"],
  ```
- Line 279: Numeric constant: 1024
  ```python
  "optimal_resolution": (1024, 1024),
  ```
- Line 279: Numeric constant: 1024
  ```python
  "optimal_resolution": (1024, 1024),
  ```
- Line 280: Numeric constant: 16
  ```python
  "resolution_multiple": 16,
  ```
- Line 289: Numeric constant: 1024
  ```python
  "optimal_resolution": (1024, 1024),
  ```
- Line 289: Numeric constant: 1024
  ```python
  "optimal_resolution": (1024, 1024),
  ```
- Line 290: Numeric constant: 16
  ```python
  "resolution_multiple": 16,
  ```
- Line 303: Numeric constant: 768
  ```python
  "optimal_resolution": (768, 768),
  ```
- Line 303: Numeric constant: 768
  ```python
  "optimal_resolution": (768, 768),
  ```
- Line 317: Numeric constant: 512
  ```python
  "optimal_resolution": (512, 512),
  ```
- Line 317: Numeric constant: 512
  ```python
  "optimal_resolution": (512, 512),
  ```
- Line 513: field_default: refiner_source
  ```python
  if not refiner_source:
  ```
- Line 587: .get() with default: 8
  ```python
  multiple = proc_config.get(
  ```
- Line 587: get_with_default: {}
  ```python
  multiple = proc_config.get(
  ```
- Line 607: Numeric constant: 512
  ```python
  optimal_width = max(optimal_width, 512)
  ```
- Line 608: Numeric constant: 512
  ```python
  optimal_height = max(optimal_height, 512)
  ```
- Line 618: .get() with default: 4096
  ```python
  max_dimension = proc_config.get(
  ```
- Line 618: get_with_default: {}
  ```python
  max_dimension = proc_config.get(
  ```
- Line 619: get_with_default: 4096
  ```python
  'diffusers_adapter', {}).get(
  ```
- Line 620: Numeric constant: 4096
  ```python
  'max_dimension', 4096)
  ```
- Line 1049: Numeric constant: 0.3
  ```python
  0.3),
  ```
- Line 1109: function_default: "canny"
  ```python
  def controlnet_inpaint(
  ```
- Line 1116: field_default: control_type
  ```python
  control_type: str = "canny",
  ```
- Line 1320: function_default: "canny"
  ```python
  def controlnet_img2img(
  ```
- Line 1326: field_default: control_type
  ```python
  control_type: str = "canny",
  ```
- Line 1489: function_default: "canny"
  ```python
  def load_controlnet(
  ```
- Line 1492: field_default: controlnet_type
  ```python
  controlnet_type: str = "canny"):
  ```
- Line 1563: function_default: "canny"
  ```python
  def generate_with_controlnet(
  ```
- Line 1568: field_default: control_type
  ```python
  control_type: str = "canny",
  ```
- Line 1585: .get() with default: ''
  ```python
  negative_prompt = negative_prompt if negative_prompt is not None else defaults.get(
  ```
- Line 1585: get_with_default: ""
  ```python
  negative_prompt = negative_prompt if negative_prompt is not None else defaults.get(
  ```
- Line 1587: .get() with default: 1.0
  ```python
  controlnet_strength = controlnet_strength if controlnet_strength is not None else defaults.get(
  ```
- Line 1589: .get() with default: 50
  ```python
  num_inference_steps = num_inference_steps if num_inference_steps is not None else defaults.get(
  ```
- Line 1589: get_with_default: 50
  ```python
  num_inference_steps = num_inference_steps if num_inference_steps is not None else defaults.get(
  ```
- Line 1590: Numeric constant: 50
  ```python
  "num_inference_steps", 50)
  ```
- Line 1591: .get() with default: 7.5
  ```python
  guidance_scale = guidance_scale if guidance_scale is not None else defaults.get(
  ```
- Line 1591: get_with_default: 7.5
  ```python
  guidance_scale = guidance_scale if guidance_scale is not None else defaults.get(
  ```
- Line 1592: Numeric constant: 7.5
  ```python
  "guidance_scale", 7.5)
  ```
- Line 1747: direct_assignment: controlnet_overhead
  ```python
  controlnet_overhead = 0
  ```

### adapters/mock_pipeline.py
Found 22 issues:

- Line 26: function_default: 7.5
  ```python
  def __call__(
  ```
- Line 31: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 31: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 32: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 32: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 33: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 33: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 51: Numeric constant: 255.0
  ```python
  mask_array = np.array(mask_image) / 255.0
  ```
- Line 51: math_operations: 255.0
  ```python
  mask_array = np.array(mask_image) / 255.0
  ```
- Line 57: Numeric constant: 1000
  ```python
  seed = abs(hash(prompt)) % 1000
  ```
- Line 64: Numeric constant: 50
  ```python
  pattern[:, :, 0] = np.random.randint(50, 100, (h, w))  # R
  ```
- Line 65: Numeric constant: 200
  ```python
  pattern[:, :, 1] = np.random.randint(100, 200, (h, w))  # G
  ```
- Line 66: Numeric constant: 150
  ```python
  pattern[:, :, 2] = np.random.randint(100, 150, (h, w))  # B
  ```
- Line 69: Numeric constant: 200
  ```python
  pattern = np.random.randint(100, 200, img_array.shape)
  ```
- Line 72: Numeric constant: 3
  ```python
  for c in range(3):
  ```
- Line 72: range_calls: 3
  ```python
  for c in range(3):
  ```
- Line 90: function_default: 0.3
  ```python
  def __call__(
  ```
- Line 91: Numeric constant: 0.3
  ```python
  self, prompt: str, image: Image.Image, strength: float = 0.3, **kwargs
  ```
- Line 91: field_default: strength
  ```python
  self, prompt: str, image: Image.Image, strength: float = 0.3, **kwargs
  ```
- Line 108: field_default: strength
  ```python
  self, prompt: str, image: Image.Image, strength: float = 0.5, **kwargs
  ```
- Line 115: Numeric constant: 10
  ```python
  noise = np.random.normal(0, strength * 10, img_array.shape)
  ```
- Line 115: math_operations: 10
  ```python
  noise = np.random.normal(0, strength * 10, img_array.shape)
  ```

### adapters/mock_pipeline_adapter.py
Found 85 issues:

- Line 25: function_default: "fp32"
  ```python
  def __init__(
  ```
- Line 27: field_default: device
  ```python
  device: str = "cpu",
  ```
- Line 28: field_default: dtype
  ```python
  dtype: str = "fp32",
  ```
- Line 60: Numeric constant: 24576
  ```python
  self.max_vram_mb = 24576 if device == "cuda" else 0  # Simulate 24GB GPU
  ```
- Line 81: function_default: 7.5
  ```python
  def generate(
  ```
- Line 85: Numeric constant: 1024
  ```python
  width: int = 1024,
  ```
- Line 85: field_default: width
  ```python
  width: int = 1024,
  ```
- Line 86: Numeric constant: 1024
  ```python
  height: int = 1024,
  ```
- Line 86: field_default: height
  ```python
  height: int = 1024,
  ```
- Line 87: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 87: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 88: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 88: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 94: Numeric constant: 50
  ```python
  f"Mock generating {width}x{height} image from prompt: {prompt[:50]}..."
  ```
- Line 98: Numeric constant: 4
  ```python
  vram_needed = (width * height * 4) / (1024 * 1024)  # Rough estimate
  ```
- Line 98: Numeric constant: 1024
  ```python
  vram_needed = (width * height * 4) / (1024 * 1024)  # Rough estimate
  ```
- Line 98: Numeric constant: 1024
  ```python
  vram_needed = (width * height * 4) / (1024 * 1024)  # Rough estimate
  ```
- Line 98: math_operations: 4
  ```python
  vram_needed = (width * height * 4) / (1024 * 1024)  # Rough estimate
  ```
- Line 98: math_operations: 1024
  ```python
  vram_needed = (width * height * 4) / (1024 * 1024)  # Rough estimate
  ```
- Line 113: Numeric constant: 32
  ```python
  np.random.seed(seed or abs(hash(prompt)) % (2**32))
  ```
- Line 113: math_operations: 32
  ```python
  np.random.seed(seed or abs(hash(prompt)) % (2**32))
  ```
- Line 121: Numeric constant: 0.05
  ```python
  noise = np.random.normal(0, 0.05, (height, width))
  ```
- Line 135: function_default: 7.5
  ```python
  def inpaint(
  ```
- Line 141: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 141: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 142: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 142: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 143: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 143: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 148: Numeric constant: 50
  ```python
  self.logger.info(f"Mock inpainting with prompt: {prompt[:50]}...")
  ```
- Line 166: function_default: 7.5
  ```python
  def img2img(
  ```
- Line 171: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 171: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 172: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 172: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 173: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 173: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 178: Numeric constant: 50
  ```python
  self.logger.info(f"Mock img2img with prompt: {prompt[:50]}...")
  ```
- Line 209: field_default: scale_factor
  ```python
  scale_factor: int = 2,
  ```
- Line 223: Numeric constant: 1.2
  ```python
  enhanced = sharpener.enhance(1.2)
  ```
- Line 240: Numeric constant: 512
  ```python
  optimal_width = max(optimal_width, 512)
  ```
- Line 241: Numeric constant: 512
  ```python
  optimal_height = max(optimal_height, 512)
  ```
- Line 244: Numeric constant: 2048
  ```python
  optimal_width = min(optimal_width, 2048)
  ```
- Line 245: Numeric constant: 2048
  ```python
  optimal_height = min(optimal_height, 2048)
  ```
- Line 254: Numeric constant: 1000
  ```python
  self.simulated_vram_mb = max(0, self.simulated_vram_mb - 1000)
  ```
- Line 254: math_operations: 1000
  ```python
  self.simulated_vram_mb = max(0, self.simulated_vram_mb - 1000)
  ```
- Line 276: function_default: 7.5
  ```python
  def controlnet_inpaint(
  ```
- Line 283: field_default: controlnet_conditioning_scale
  ```python
  controlnet_conditioning_scale: float = 1.0,
  ```
- Line 284: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 284: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 285: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 285: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 286: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 286: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 299: function_default: 7.5
  ```python
  def controlnet_img2img(
  ```
- Line 305: field_default: controlnet_conditioning_scale
  ```python
  controlnet_conditioning_scale: float = 1.0,
  ```
- Line 306: Numeric constant: 0.8
  ```python
  strength: float = 0.8,
  ```
- Line 306: field_default: strength
  ```python
  strength: float = 0.8,
  ```
- Line 307: Numeric constant: 50
  ```python
  num_inference_steps: int = 50,
  ```
- Line 307: field_default: num_inference_steps
  ```python
  num_inference_steps: int = 50,
  ```
- Line 308: Numeric constant: 7.5
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 308: field_default: guidance_scale
  ```python
  guidance_scale: float = 7.5,
  ```
- Line 337: Numeric constant: 4000
  ```python
  "generate": 4000,
  ```
- Line 338: Numeric constant: 3500
  ```python
  "inpaint": 3500,
  ```
- Line 339: Numeric constant: 3000
  ```python
  "img2img": 3000,
  ```
- Line 340: Numeric constant: 2000
  ```python
  "refine": 2000,
  ```
- Line 341: Numeric constant: 1500
  ```python
  "enhance": 1500,
  ```
- Line 344: .get() with default: 2000
  ```python
  base = estimates.get(operation, 2000)
  ```
- Line 344: Numeric constant: 2000
  ```python
  base = estimates.get(operation, 2000)
  ```
- Line 344: get_with_default: 2000
  ```python
  base = estimates.get(operation, 2000)
  ```
- Line 349: Numeric constant: 1024
  ```python
  base_pixels = 1024 * 1024
  ```
- Line 349: Numeric constant: 1024
  ```python
  base_pixels = 1024 * 1024
  ```
- Line 349: math_operations: 1024
  ```python
  base_pixels = 1024 * 1024
  ```
- Line 359: Numeric constant: 0.7
  ```python
  self.simulated_vram_mb = self.simulated_vram_mb * 0.7
  ```
- Line 359: math_operations: 0.7
  ```python
  self.simulated_vram_mb = self.simulated_vram_mb * 0.7
  ```
- Line 451: Numeric constant: 40
  ```python
  watermark_h, watermark_w = 40, 120
  ```
- Line 451: Numeric constant: 120
  ```python
  watermark_h, watermark_w = 40, 120
  ```
- Line 460: Numeric constant: 10
  ```python
  img_array[10:30, 10:110] = np.clip(
  ```
- Line 460: Numeric constant: 30
  ```python
  img_array[10:30, 10:110] = np.clip(
  ```
- Line 460: Numeric constant: 10
  ```python
  img_array[10:30, 10:110] = np.clip(
  ```
- Line 460: Numeric constant: 110
  ```python
  img_array[10:30, 10:110] = np.clip(
  ```
- Line 461: Numeric constant: 10
  ```python
  img_array[10:30, 10:110] + 100, 0, 255)
  ```
- Line 461: Numeric constant: 30
  ```python
  img_array[10:30, 10:110] + 100, 0, 255)
  ```
- Line 461: Numeric constant: 10
  ```python
  img_array[10:30, 10:110] + 100, 0, 255)
  ```
- Line 461: Numeric constant: 110
  ```python
  img_array[10:30, 10:110] + 100, 0, 255)
  ```

### cli/args.py
Found 34 issues:

- Line 60: Numeric constant: 1280
  ```python
  "720p": (1280, 720),
  ```
- Line 60: Numeric constant: 720
  ```python
  "720p": (1280, 720),
  ```
- Line 61: Numeric constant: 1920
  ```python
  "1080p": (1920, 1080),
  ```
- Line 61: Numeric constant: 1080
  ```python
  "1080p": (1920, 1080),
  ```
- Line 62: Numeric constant: 2560
  ```python
  "1440p": (2560, 1440),
  ```
- Line 62: Numeric constant: 1440
  ```python
  "1440p": (2560, 1440),
  ```
- Line 63: Numeric constant: 3840
  ```python
  "4k": (3840, 2160),
  ```
- Line 63: Numeric constant: 2160
  ```python
  "4k": (3840, 2160),
  ```
- Line 64: Numeric constant: 5120
  ```python
  "5k": (5120, 2880),
  ```
- Line 64: Numeric constant: 2880
  ```python
  "5k": (5120, 2880),
  ```
- Line 65: Numeric constant: 7680
  ```python
  "8k": (7680, 4320),
  ```
- Line 65: Numeric constant: 4320
  ```python
  "8k": (7680, 4320),
  ```
- Line 67: Numeric constant: 1920
  ```python
  "16:9": (1920, 1080),
  ```
- Line 67: Numeric constant: 1080
  ```python
  "16:9": (1920, 1080),
  ```
- Line 68: Numeric constant: 2560
  ```python
  "21:9": (2560, 1080),
  ```
- Line 68: Numeric constant: 1080
  ```python
  "21:9": (2560, 1080),
  ```
- Line 69: Numeric constant: 3840
  ```python
  "32:9": (3840, 1080),
  ```
- Line 69: Numeric constant: 1080
  ```python
  "32:9": (3840, 1080),
  ```
- Line 70: Numeric constant: 1080
  ```python
  "9:16": (1080, 1920),  # Portrait
  ```
- Line 70: Numeric constant: 1920
  ```python
  "9:16": (1080, 1920),  # Portrait
  ```
- Line 72: Numeric constant: 3440
  ```python
  "ultrawide": (3440, 1440),
  ```
- Line 72: Numeric constant: 1440
  ```python
  "ultrawide": (3440, 1440),
  ```
- Line 73: Numeric constant: 5120
  ```python
  "superultrawide": (5120, 1440),
  ```
- Line 73: Numeric constant: 1440
  ```python
  "superultrawide": (5120, 1440),
  ```
- Line 74: Numeric constant: 2160
  ```python
  "portrait4k": (2160, 3840),
  ```
- Line 74: Numeric constant: 3840
  ```python
  "portrait4k": (2160, 3840),
  ```
- Line 75: Numeric constant: 3840
  ```python
  "square4k": (3840, 3840),
  ```
- Line 75: Numeric constant: 3840
  ```python
  "square4k": (3840, 3840),
  ```
- Line 77: Numeric constant: 1170
  ```python
  "iphone": (1170, 2532),
  ```
- Line 77: Numeric constant: 2532
  ```python
  "iphone": (1170, 2532),
  ```
- Line 78: Numeric constant: 2048
  ```python
  "ipad": (2048, 2732),
  ```
- Line 78: Numeric constant: 2732
  ```python
  "ipad": (2048, 2732),
  ```
- Line 79: Numeric constant: 1440
  ```python
  "android": (1440, 3200),
  ```
- Line 79: Numeric constant: 3200
  ```python
  "android": (1440, 3200),
  ```

### cli/main.py
Found 11 issues:

- Line 268: direct_assignment: success_count
  ```python
  success_count = 0
  ```
- Line 313: Numeric constant: 32
  ```python
  seed = abs(hash(f"{input_path}_{datetime.now()}")) % (2**32)
  ```
- Line 313: math_operations: 32
  ```python
  seed = abs(hash(f"{input_path}_{datetime.now()}")) % (2**32)
  ```
- Line 350: Numeric constant: 5
  ```python
  if i % 5 == 0:
  ```
- Line 360: Numeric constant: 50
  ```python
  logger.info(f"\n{'=' * 50}")
  ```
- Line 360: math_operations: 50
  ```python
  logger.info(f"\n{'=' * 50}")
  ```
- Line 369: Numeric constant: 130
  ```python
  return 130
  ```
- Line 390: Numeric constant: 3
  ```python
  return 3
  ```
- Line 395: Numeric constant: 4
  ```python
  return 4
  ```
- Line 400: Numeric constant: 5
  ```python
  return 5
  ```
- Line 407: Numeric constant: 6
  ```python
  return 6
  ```

### cli/setup_wizard.py
Found 19 issues:

- Line 70: Numeric constant: 60
  ```python
  self.logger.info("\n" + "=" * 60)
  ```
- Line 70: math_operations: 60
  ```python
  self.logger.info("\n" + "=" * 60)
  ```
- Line 72: Numeric constant: 60
  ```python
  self.logger.info("=" * 60)
  ```
- Line 72: math_operations: 60
  ```python
  self.logger.info("=" * 60)
  ```
- Line 120: math_operations: 3
  ```python
  model_id="stabilityai/stable-diffusion-3-medium",
  ```
- Line 192: Numeric constant: 1024
  ```python
  min_val=1024,
  ```
- Line 193: Numeric constant: 49152
  ```python
  max_val=49152,
  ```
- Line 232: math_operations: 2.0
  ```python
  "LoRA weight (0.1-2.0, default 1.0): ",
  ```
- Line 234: Numeric constant: 0.1
  ```python
  min_val=0.1,
  ```
- Line 235: Numeric constant: 2.0
  ```python
  max_val=2.0,
  ```
- Line 328: Numeric constant: 60
  ```python
  self.logger.info("\n" + "=" * 60)
  ```
- Line 328: math_operations: 60
  ```python
  self.logger.info("\n" + "=" * 60)
  ```
- Line 330: Numeric constant: 60
  ```python
  self.logger.info("=" * 60)
  ```
- Line 330: math_operations: 60
  ```python
  self.logger.info("=" * 60)
  ```
- Line 340: field_default: default
  ```python
  def _confirm(self, prompt: str, default: bool = False) -> bool:
  ```
- Line 363: function_default: "auto"
  ```python
  def _get_path(
  ```
- Line 364: field_default: must_exist
  ```python
  self, prompt: str, must_exist: bool = False, create: bool = False,
  ```
- Line 364: field_default: create
  ```python
  self, prompt: str, must_exist: bool = False, create: bool = False,
  ```
- Line 365: field_default: path_type
  ```python
  path_type: str = "auto"
  ```

### cli/utils.py
Found 19 issues:

- Line 82: Numeric constant: 60
  ```python
  logger.info("\n" + "=" * 60)
  ```
- Line 82: math_operations: 60
  ```python
  logger.info("\n" + "=" * 60)
  ```
- Line 84: Numeric constant: 60
  ```python
  logger.info("=" * 60)
  ```
- Line 84: math_operations: 60
  ```python
  logger.info("=" * 60)
  ```
- Line 126: Numeric constant: 60
  ```python
  logger.info("=" * 60)
  ```
- Line 126: math_operations: 60
  ```python
  logger.info("=" * 60)
  ```
- Line 148: Numeric constant: 50
  ```python
  logger.info("=" * 50)
  ```
- Line 148: math_operations: 50
  ```python
  logger.info("=" * 50)
  ```
- Line 169: field_default: enabled
  ```python
  if model_config and model_config.enabled:
  ```
- Line 180: field_default: is_valid
  ```python
  if not is_valid:
  ```
- Line 191: field_default: exists
  ```python
  if not exists:
  ```
- Line 202: Numeric constant: 1024
  ```python
  vram_mb = vram_bytes / (1024 * 1024)
  ```
- Line 202: Numeric constant: 1024
  ```python
  vram_mb = vram_bytes / (1024 * 1024)
  ```
- Line 202: math_operations: 1024
  ```python
  vram_mb = vram_bytes / (1024 * 1024)
  ```
- Line 207: Numeric constant: 1024
  ```python
  allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
  ```
- Line 207: Numeric constant: 1024
  ```python
  allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
  ```
- Line 207: math_operations: 1024
  ```python
  allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
  ```
- Line 218: Numeric constant: 50
  ```python
  logger.info("\n" + "=" * 50)
  ```
- Line 218: math_operations: 50
  ```python
  logger.info("\n" + "=" * 50)
  ```

### config/lora_manager.py
Found 4 issues:

- Line 174: Numeric constant: 1.5
  ```python
  if total_weight > 1.5:
  ```
- Line 174: if_comparison: 1.5
  ```python
  if total_weight > 1.5:
  ```
- Line 179: Numeric constant: 1.5
  ```python
  scale_factor = 1.5 / total_weight
  ```
- Line 282: direct_assignment: additional_steps
  ```python
  
  ```

### config/pipeline_config.py
Found 11 issues:

- Line 34: field_default: save_stages
  ```python
  save_stages: bool = False,
  ```
- Line 36: field_default: verbose
  ```python
  verbose: bool = False,
  ```
- Line 76: get_with_default: user_config.max_vram_usage_mb
  ```python
  vram_limit_mb=kwargs.get(
  ```
- Line 79: .get() with default: 'aggressive'
  ```python
  enable_artifacts_check=kwargs.get(
  ```
- Line 79: get_with_default: "aggressive"
  ```python
  enable_artifacts_check=kwargs.get(
  ```
- Line 83: Numeric constant: 0.1
  ```python
  0.1 if user_config.auto_artifact_detection else 1.0),
  ```
- Line 122: function_default: "auto"
  ```python
  def create_adapter(
  ```
- Line 125: field_default: adapter_type
  ```python
  adapter_type: str = "auto") -> Any:
  ```
- Line 159: field_default: model_id
  ```python
  if model_config.model_id:
  ```
- Line 164: field_default: else
  ```python
  else:
  ```
- Line 202: get_with_default: torch.float32
  ```python
  return dtype_map.get(dtype_str, torch.float32)
  ```

### config/user_config.py
Found 8 issues:

- Line 25: field_default: dtype
  ```python
  dtype: str = "fp16"
  ```
- Line 29: field_default: enabled
  ```python
  enabled: bool = True
  ```
- Line 152: get_with_default: f"lora_{i}"
  ```python
  lora_name = lora_data.get("name", f"lora_{i}")
  ```
- Line 323: Numeric constant: 120
  ```python
  width=120,
  ```
- Line 372: math_operations: 3
  ```python
  model_id="stabilityai/stable-diffusion-3-medium",
  ```
- Line 493: math_operations: 3
  ```python
  model_id: "stabilityai/stable-diffusion-3-medium"
  ```
- Line 557: direct_assignment: clear_cache_frequency
  ```python
  clear_cache_frequency: 5  # Clear CUDA cache every N operations
  ```
- Line 561: direct_assignment: output_compression
  ```python
  output_compression: 0  # PNG compression level (0-9)
  ```

### core/boundary_tracker.py
Found 7 issues:

- Line 49: function_default: "unknown"
  ```python
  def add_boundary(
  ```
- Line 57: field_default: method
  ```python
  method: str = "unknown",
  ```
- Line 145: function_default: "progressive"
  ```python
  def add_progressive_boundaries(
  ```
- Line 150: field_default: method
  ```python
  method: str = "progressive",
  ```
- Line 238: function_default: 10
  ```python
  def get_boundary_regions(
  ```
- Line 239: Numeric constant: 10
  ```python
  self, width: int, height: int, padding: int = 10
  ```
- Line 239: field_default: padding
  ```python
  self, width: int, height: int, padding: int = 10
  ```

### core/config.py
Found 3 issues:

- Line 140: get_with_default: yaml_field_name
  ```python
  field_name = field_mapping.get(yaml_field_name, yaml_field_name)
  ```
- Line 156: get_with_default: yaml_field_name
  ```python
  field_name = field_mapping.get(yaml_field_name, yaml_field_name)
  ```
- Line 387: get_with_default: effective
  ```python
  return self.strategy_map.get(effective, effective)
  ```

### core/configuration_manager.py
Found 11 issues:

- Line 130: get_with_default: {}
  ```python
  self._schema_resolver = RefResolver(base_uri, self._schemas.get('base_schema', {}))
  ```
- Line 228: .get() with default: ''
  ```python
  env_path = os.environ.get("EXPANDOR_CONFIG_PATH", "")
  ```
- Line 228: get_with_default: ""
  ```python
  env_path = os.environ.get("EXPANDOR_CONFIG_PATH", "")
  ```
- Line 263: .get() with default: ''
  ```python
  env_path = os.environ.get("EXPANDOR_CONFIG_PATH", "")
  ```
- Line 263: get_with_default: ""
  ```python
  env_path = os.environ.get("EXPANDOR_CONFIG_PATH", "")
  ```
- Line 285: Numeric constant: 9
  ```python
  config_path = key[9:].lower().replace("_", ".")
  ```
- Line 399: get_with_default: {}
  ```python
  f"Available strategies: {list(self._config_cache.get('strategies', {}).keys())}"
  ```
- Line 410: get_with_default: {}
  ```python
  f"Available processors: {list(self._config_cache.get('processors', {}).keys())}"
  ```
- Line 413: field_default: create
  ```python
  def get_path(self, path_key: str, create: bool = True,
  ```
- Line 413: function_default: "directory"
  ```python
  def get_path(self, path_key: str, create: bool = True,
  ```
- Line 414: field_default: path_type
  ```python
  path_type: str = "directory") -> Path:
  ```

### core/exceptions.py
Found 2 issues:

- Line 28: function_default: ""
  ```python
  def __init__(
  ```
- Line 33: field_default: message
  ```python
  message: str = ""):
  ```

### core/expandor.py
Found 13 issues:

- Line 311: Numeric constant: 65536
  ```python
  if target_w > 65536 or target_h > 65536:
  ```
- Line 311: Numeric constant: 65536
  ```python
  if target_w > 65536 or target_h > 65536:
  ```
- Line 311: if_comparison: 65536
  ```python
  if target_w > 65536 or target_h > 65536:
  ```
- Line 465: Numeric constant: 1024
  ```python
  free_before = torch.cuda.mem_get_info()[0] / (1024**2)
  ```
- Line 466: Numeric constant: 1024
  ```python
  total_vram = torch.cuda.mem_get_info()[1] / (1024**2)
  ```
- Line 488: Numeric constant: 1024
  ```python
  free_after = torch.cuda.mem_get_info()[0] / (1024**2)
  ```
- Line 568: Numeric constant: 10
  ```python
  reduced_metadata['artifact_repair_steps'] = max(10, normal_steps // 2)  # Half steps, min 10
  ```
- Line 569: Numeric constant: 0.2
  ```python
  reduced_metadata['artifact_repair_strength'] = max(0.2, normal_strength * 0.6)  # 60% strength, min 0.2
  ```
- Line 569: Numeric constant: 0.6
  ```python
  reduced_metadata['artifact_repair_strength'] = max(0.2, normal_strength * 0.6)  # 60% strength, min 0.2
  ```
- Line 686: Numeric constant: 60
  ```python
  self.logger.error("=" * 60)
  ```
- Line 686: math_operations: 60
  ```python
  self.logger.error("=" * 60)
  ```
- Line 697: Numeric constant: 60
  ```python
  self.logger.error("=" * 60)
  ```
- Line 697: math_operations: 60
  ```python
  self.logger.error("=" * 60)
  ```

### core/metadata_tracker.py
Found 6 issues:

- Line 31: Numeric constant: 1000
  ```python
  self.operation_id = f"op_{int(time.time() * 1000)}"
  ```
- Line 31: math_operations: 1000
  ```python
  self.operation_id = f"op_{int(time.time() * 1000)}"
  ```
- Line 73: .get() with default: 'unknown'
  ```python
  "model_type": config.source_metadata.get("model", "unknown"),
  ```
- Line 73: get_with_default: "unknown"
  ```python
  "model_type": config.source_metadata.get("model", "unknown"),
  ```
- Line 115: field_default: success
  ```python
  def exit_stage(self, success: bool = True, error: Optional[str] = None):
  ```
- Line 133: .get() with default: 0
  ```python
  "duration": self.stage_timings.get(self.current_stage, 0),
  ```

### core/pipeline_orchestrator.py
Found 12 issues:

- Line 103: direct_assignment: fallback_count
  ```python
  fallback_count = 0
  ```
- Line 321: Numeric constant: 1024
  ```python
  "tiled_large": ("tiled_expansion", {"tile_size": 1024}),
  ```
- Line 322: Numeric constant: 768
  ```python
  "tiled_medium": ("tiled_expansion", {"tile_size": 768}),
  ```
- Line 323: Numeric constant: 512
  ```python
  "tiled_small": ("tiled_expansion", {"tile_size": 512}),
  ```
- Line 357: get_with_default: {}
  ```python
  global_defaults = self.config.get("global_defaults", {})
  ```
- Line 360: get_with_default: {}
  ```python
  strategies_config = self.config.get("strategies", {})
  ```
- Line 361: get_with_default: {}
  ```python
  strategy_defaults = strategies_config.get(strategy_name, {})
  ```
- Line 364: get_with_default: {}
  ```python
  user_overrides = self.config.get("user_overrides", {})
  ```
- Line 365: get_with_default: {}
  ```python
  user_config = user_overrides.get(strategy_name, {})
  ```
- Line 462: get_with_default: []
  ```python
  stages=raw_result.get("stages", []),  # Optional - stages may be empty
  ```
- Line 463: get_with_default: []
  ```python
  boundaries=raw_result.get("boundaries", []),  # Optional - boundaries may be empty
  ```
- Line 465: get_with_default: {}
  ```python
  metadata=raw_result.get("metadata", {}),  # Optional - metadata may be empty
  ```

### core/result.py
Found 11 issues:

- Line 19: field_default: vram_used_mb
  ```python
  vram_used_mb: float = 0.0
  ```
- Line 20: field_default: artifacts_detected
  ```python
  artifacts_detected: int = 0
  ```
- Line 44: field_default: success
  ```python
  success: bool = True
  ```
- Line 51: field_default: seams_detected
  ```python
  seams_detected: int = 0
  ```
- Line 52: field_default: artifacts_fixed
  ```python
  artifacts_fixed: int = 0
  ```
- Line 53: field_default: refinement_passes
  ```python
  refinement_passes: int = 0
  ```
- Line 54: field_default: quality_score
  ```python
  quality_score: float = 1.0
  ```
- Line 57: field_default: vram_peak_mb
  ```python
  vram_peak_mb: float = 0.0
  ```
- Line 58: field_default: total_duration_seconds
  ```python
  total_duration_seconds: float = 0.0
  ```
- Line 59: field_default: strategy_used
  ```python
  strategy_used: str = ""
  ```
- Line 60: field_default: fallback_count
  ```python
  fallback_count: int = 0
  ```

### core/strategy_selector.py
Found 9 issues:

- Line 122: get_with_default: user_strategy
  ```python
  mapped_strategy = strategy_mapping.get(
  ```
- Line 144: field_default: dry_run
  ```python
  def select(self, config, dry_run: bool = False) -> BaseExpansionStrategy:
  ```
- Line 217: .get() with default: 'unknown'
  ```python
  model_type = config.source_metadata.get("model", "unknown")
  ```
- Line 217: get_with_default: "unknown"
  ```python
  model_type = config.source_metadata.get("model", "unknown")
  ```
- Line 322: field_default: swpo_config
  ```python
  if "enabled" not in swpo_config:
  ```
- Line 338: Numeric constant: 1.1
  ```python
  aspect_tolerance = 1.1  # This is a tolerance value, kept as algorithmic constant
  ```
- Line 338: direct_assignment: aspect_tolerance
  ```python
  aspect_tolerance = 1.1  # This is a tolerance value, kept as algorithmic constant
  ```
- Line 340: Numeric constant: 4
  ```python
  massive_expansion = moderate_expansion * 4  # 16 if moderate is 4
  ```
- Line 358: field_default: prog_config
  ```python
  if "enabled" not in prog_config:
  ```

### core/vram_manager.py
Found 12 issues:

- Line 65: math_operations: 76
  ```python
  Copy implementation from lines 22-76 of vram_calculator.py
  ```
- Line 137: math_operations: 92
  ```python
  """Get available VRAM in MB - from lines 78-92"""
  ```
- Line 217: Numeric constant: 4
  ```python
  overlap = max(min_overlap, tile_size // 4)
  ```
- Line 217: math_operations: 4
  ```python
  overlap = max(min_overlap, tile_size // 4)
  ```
- Line 252: field_default: pipeline_type
  ```python
  self, pipeline_type: str = "sdxl", include_vae: bool = True
  ```
- Line 252: field_default: include_vae
  ```python
  self, pipeline_type: str = "sdxl", include_vae: bool = True
  ```
- Line 337: Numeric constant: 1000000
  ```python
  mb_per_pixel = mb_per_megapixel / 1_000_000
  ```
- Line 356: Numeric constant: 64
  ```python
  tile_size = (tile_size // 64) * 64
  ```
- Line 356: Numeric constant: 64
  ```python
  tile_size = (tile_size // 64) * 64
  ```
- Line 356: math_operations: 64
  ```python
  tile_size = (tile_size // 64) * 64
  ```
- Line 356: math_operations: 64
  ```python
  tile_size = (tile_size // 64) * 64
  ```
- Line 381: direct_assignment: available
  ```python
  available = 0  # CPU mode
  ```

### processors/artifact_detector_enhanced.py
Found 8 issues:

- Line 21: direct_assignment: NONE
  ```python
  
  ```
- Line 23: direct_assignment: LOW
  ```python
  LOW = 1
  ```
- Line 24: direct_assignment: MEDIUM
  ```python
  MEDIUM = 2
  ```
- Line 25: Numeric constant: 3
  ```python
  HIGH = 3
  ```
- Line 25: direct_assignment: HIGH
  ```python
  HIGH = 3
  ```
- Line 26: Numeric constant: 4
  ```python
  CRITICAL = 4
  ```
- Line 26: direct_assignment: CRITICAL
  ```python
  CRITICAL = 4
  ```
- Line 246: Numeric constant: 5.0
  ```python
  if severity == ArtifactSeverity.CRITICAL and artifact_percentage > 5.0:
  ```

### processors/artifact_removal.py
Found 11 issues:

- Line 38: math_operations: 115
  ```python
  Adapted from lines 22-115 of smart_detector.py
  ```
- Line 59: direct_assignment: seam_count
  ```python
  seam_count = 0
  ```
- Line 62: get_with_default: []
  ```python
  boundaries = metadata.get("progressive_boundaries", [])
  ```
- Line 63: get_with_default: []
  ```python
  boundaries_v = metadata.get("progressive_boundaries_vertical", [])
  ```
- Line 64: get_with_default: []
  ```python
  seam_details = metadata.get("seam_details", [])
  ```
- Line 84: .get() with default: False
  ```python
  if metadata.get("used_tiled", False):
  ```
- Line 85: get_with_default: []
  ```python
  tile_boundaries = metadata.get("tile_boundaries", [])
  ```
- Line 113: Numeric constant: 255.0
  ```python
  problem_mask = np.array(mask_pil) / 255.0
  ```
- Line 113: math_operations: 255.0
  ```python
  problem_mask = np.array(mask_pil) / 255.0
  ```
- Line 205: Numeric constant: 3
  ```python
  grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  ```
- Line 206: Numeric constant: 3
  ```python
  grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  ```

### processors/boundary_analysis.py
Found 6 issues:

- Line 267: get_with_default: severity_values['default']
            
  ```python
  severity_value = severity_values.get(
  ```
- Line 315: .get() with default: 0
  ```python
  issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
  ```
- Line 316: .get() with default: 0
  ```python
  severities[issue["severity"]] = severities.get(
  ```
- Line 320: .get() with default: 0
  ```python
  if issue_types.get("color_discontinuity", 0) > 0:
  ```
- Line 348: get_with_default: {}
  ```python
  if "color_difference" in issue.get("metrics", {})
  ```
- Line 369: Numeric constant: 3
  ```python
  rgb_map = np.zeros((height, width, 3), dtype=np.uint8)
  ```

### processors/controlnet_extractors.py
Found 9 issues:

- Line 113: field_default: dilate
  ```python
  dilate: bool = True,
  ```
- Line 114: field_default: l2_gradient
  ```python
  l2_gradient: bool = False
  ```
- Line 133: get_with_default: {}
  ```python
  canny_config = self.extractor_config.get("canny", {})
  ```
- Line 137: get_with_default: self.processor_config['canny']['default_low_threshold']
  ```python
  low_threshold = canny_config.get("default_low_threshold",
  ```
- Line 140: get_with_default: self.processor_config['canny']['default_high_threshold']
  ```python
  high_threshold = canny_config.get("default_high_threshold",
  ```
- Line 228: function_default: "gaussian"
  ```python
  def extract_blur(
  ```
- Line 232: field_default: blur_type
  ```python
  blur_type: str = "gaussian"
  ```
- Line 249: get_with_default: {}
  ```python
  blur_config = self.extractor_config.get("blur", {})
  ```
- Line 347: get_with_default: {}
  ```python
  depth_config = self.extractor_config.get("depth", {})
  ```

### processors/edge_analysis.py
Found 10 issues:

- Line 19: field_default: ImportError
  ```python
  except ImportError:
  ```
- Line 138: Numeric constant: 3
  ```python
  sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
  ```
- Line 139: Numeric constant: 3
  ```python
  sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
  ```
- Line 319: direct_assignment: seam_impact
  ```python
  seam_impact = 0.0
  ```
- Line 323: direct_assignment: artifact_impact
  ```python
  
  ```
- Line 339: function_default: "vertical"
  ```python
  def detect_color_discontinuity(
  ```
- Line 344: field_default: direction
  ```python
  direction: str = "vertical",
  ```
- Line 399: function_default: 5
  ```python
  def create_edge_mask(
  ```
- Line 400: Numeric constant: 5
  ```python
  self, edges: List[EdgeInfo], image_size: Tuple[int, int], dilation: int = 5
  ```
- Line 400: field_default: dilation
  ```python
  self, edges: List[EdgeInfo], image_size: Tuple[int, int], dilation: int = 5
  ```

### processors/quality_orchestrator.py
Found 3 issues:

- Line 135: direct_assignment: refinement_passes
  ```python
  refinement_passes = 0
  ```
- Line 285: get_with_default: self.processor_config['default_quality_score']
  ```python
  score = boundary_analysis.get("quality_score", self.processor_config['default_quality_score'])
  ```
- Line 296: get_with_default: self.processor_config['default_severity_penalty']
  ```python
  score -= severity_penalty.get(detection_result.severity, self.processor_config['default_severity_penalty'])
  ```

### processors/quality_validator.py
Found 6 issues:

- Line 50: get_with_default: {}
  ```python
  self.detection_config = config.get("quality_validation", {}).get(
  ```
- Line 50: get_with_default: {}
        
  ```python
  self.detection_config = config.get("quality_validation", {}).get(
  ```
- Line 54: function_default: "aggressive"
  ```python
  def validate(
  ```
- Line 58: field_default: detection_level
  ```python
  detection_level: str = "aggressive",
  ```
- Line 75: get_with_default: self.detection_config.get("aggressive", {}
  ```python
  level_config = self.detection_config.get(
  ```
- Line 151: .get() with default: 0.0
  ```python
  score -= severity_penalties.get(severity, 0.0)
  ```

### processors/refinement/smart_refiner.py
Found 3 issues:

- Line 118: field_default: save_stages
  ```python
  save_stages: bool = False,
  ```
- Line 163: direct_assignment: iterations
  ```python
  iterations = 0
  ```
- Line 164: direct_assignment: total_improvement
  ```python
  total_improvement = 0.0
  ```

### processors/seam_repair.py
Found 2 issues:

- Line 133: Numeric constant: 1000
  ```python
  timestamp = int(time.time() * 1000)
  ```
- Line 133: math_operations: 1000
  ```python
  timestamp = int(time.time() * 1000)
  ```

### processors/tiled_processor.py
Found 4 issues:

- Line 119: direct_assignment: tile_idx
  ```python
  tile_idx = 0
  ```
- Line 125: direct_assignment: y
  ```python
  y = 0
  ```
- Line 130: direct_assignment: x
  ```python
  
  ```
- Line 196: field_default: save_tiles
  ```python
  save_tiles: bool = False,
  ```

### strategies/base_strategy.py
Found 9 issues:

- Line 66: field_default: strategy_name
  ```python
  if 'progressiveoutpaint' in strategy_name:
  ```
- Line 68: field_default: strategy_name
  ```python
  elif 'cpuoffload' in strategy_name:
  ```
- Line 70: field_default: strategy_name
  ```python
  elif 'tiledexpansion' in strategy_name:
  ```
- Line 142: Numeric constant: 1.2
  ```python
  "peak_vram_mb": base_req * 1.2,  # 20% buffer
  ```
- Line 225: Numeric constant: 1000
  ```python
  timestamp = int(time.time() * 1000)
  ```
- Line 225: math_operations: 1000
  ```python
  timestamp = int(time.time() * 1000)
  ```
- Line 265: Numeric constant: 5
  ```python
  def _cleanup_temp_files(self, keep_last: int = 5):
  ```
- Line 265: field_default: keep_last
  ```python
  def _cleanup_temp_files(self, keep_last: int = 5):
  ```
- Line 265: function_default: 5
  ```python
  def _cleanup_temp_files(self, keep_last: int = 5):
  ```

### strategies/controlnet_progressive.py
Found 4 issues:

- Line 85: .get() with default: True
  ```python
  self.extract_at_each_step = self.controlnet_config.get(
  ```
- Line 85: get_with_default: True  # Default to True for ControlNet
        
  ```python
  self.extract_at_each_step = self.controlnet_config.get(
  ```
- Line 241: get_with_default: self.strategy_config.get('default_strength', self.strategy_config['strength_fallback']
  ```python
  base_strength = kwargs.get(
  ```
- Line 261: get_with_default: {}
  ```python
  controlnet_params.update(kwargs.get("controlnet_params", {}))
  ```

### strategies/cpu_offload.py
Found 8 issues:

- Line 154: get_with_default: {}
  ```python
  self._context.get("pipeline_registry", {}).keys()
  ```
- Line 231: get_with_default: optimal_tile_size
  ```python
  "tile_size": stage.get("tile_size", optimal_tile_size),
  ```
- Line 332: get_with_default: {}
  ```python
  steps = self.config.get(
  ```
- Line 334: get_with_default: self.strategy_config['aspect_adjust_steps']
  ```python
  {}).get(
  ```
- Line 445: get_with_default: {}
  ```python
  num_inference_steps=self.config.get('parameters', {}).get(
  ```
- Line 445: get_with_default: self.strategy_config['tile_generation_steps']
  ```python
  num_inference_steps=self.config.get('parameters', {}).get(
  ```
- Line 506: get_with_default: {}
  ```python
  num_inference_steps=self.config.get(
  ```
- Line 508: get_with_default: self.strategy_config['tile_refinement_steps']
  ```python
  {}).get(
  ```

### strategies/direct_upscale.py
Found 1 issues:

- Line 481: Numeric constant: 3
  ```python
  "RealESRGAN_x4plus" if pass_scale >= 3 else "RealESRGAN_x2plus"),
  ```

### strategies/experimental/hybrid_adaptive.py
Found 5 issues:

- Line 437: Numeric constant: 1.3
  ```python
  "config_overrides": {"max_expansion_ratio": 1.3},
  ```
- Line 460: Numeric constant: 384
  ```python
  "config_overrides": {"tile_size": 384, "overlap": 64},
  ```
- Line 460: Numeric constant: 64
  ```python
  "config_overrides": {"tile_size": 384, "overlap": 64},
  ```
- Line 463: Numeric constant: 512
  ```python
  estimated_vram=512,
  ```
- Line 464: Numeric constant: 0.7
  ```python
  estimated_quality=0.7,
  ```

### strategies/progressive_outpaint.py
Found 14 issues:

- Line 71: math_operations: 123
  ```python
  Copy from lines 81-123 of aspect_adjuster.py
  ```
- Line 102: Numeric constant: 3
  ```python
  pixels = sample.reshape(-1, 3)
  ```
- Line 263: math_operations: 698
  ```python
  Adapted from _execute_outpaint_step lines 524-698
  ```
- Line 366: math_operations: 625
  ```python
  Adapted from lines 574-625 of aspect_adjuster.py
  ```
- Line 399: Numeric constant: 3
  ```python
  3)
  ```
- Line 409: Numeric constant: 3
  ```python
  3)
  ```
- Line 419: Numeric constant: 3
  ```python
  3)
  ```
- Line 427: Numeric constant: 3
  ```python
  3)
  ```
- Line 473: .get() with default: 'progressive'
  ```python
  step_type = step_info.get("step_type", "progressive")
  ```
- Line 473: get_with_default: "progressive"
  ```python
  step_type = step_info.get("step_type", "progressive")
  ```
- Line 503: .get() with default: 'progressive'
  ```python
  step_type = step_info.get("step_type", "progressive")
  ```
- Line 503: get_with_default: "progressive"
  ```python
  step_type = step_info.get("step_type", "progressive")
  ```
- Line 630: .get() with default: 'both'
  ```python
  direction = step_info.get("direction", "both")
  ```
- Line 630: get_with_default: "both"
  ```python
  direction = step_info.get("direction", "both")
  ```

### strategies/strategy_selector.py
Found 4 issues:

- Line 49: field_default: quality_priority
  ```python
  quality_priority: bool = True,
  ```
- Line 200: function_default: 50
  ```python
  def register_strategy(
  ```
- Line 204: Numeric constant: 50
  ```python
  priority: int = 50):
  ```
- Line 204: field_default: priority
  ```python
  priority: int = 50):
  ```

### strategies/swpo_strategy.py
Found 16 issues:

- Line 138: Numeric constant: 1024
  ```python
  window_vram_mb = (effective_area * channels * bytes_per_pixel * batch_size * safety_factor) / (1024**2)
  ```
- Line 190: get_with_default: {}
  ```python
  self._context.get("pipeline_registry", {}).keys()
  ```
- Line 454: direct_assignment: window_index
  ```python
  window_index = 0
  ```
- Line 475: direct_assignment: x1
  ```python
  x1 = 0
  ```
- Line 513: direct_assignment: y1
  ```python
  y1 = 0
  ```
- Line 571: direct_assignment: paste_y
  ```python
  paste_y = 0
  ```
- Line 594: direct_assignment: paste_x
  ```python
  paste_x = 0
  ```
- Line 806: get_with_default: config.num_inference_steps
  ```python
  num_inference_steps=self.strategy_params.get(
  ```
- Line 808: get_with_default: config.guidance_scale
  ```python
  guidance_scale=self.strategy_params.get(
  ```
- Line 890: function_default: 0.02
  ```python
  def _add_noise_to_mask(
  ```
- Line 891: Numeric constant: 0.02
  ```python
  self, image: Image.Image, mask: Image.Image, strength: float = 0.02
  ```
- Line 891: field_default: strength
  ```python
  self, image: Image.Image, mask: Image.Image, strength: float = 0.02
  ```
- Line 994: Numeric constant: 128
  ```python
  mask_array > 128, iterations=kernel_size // 2)
  ```
- Line 996: Numeric constant: 128
  ```python
  mask_array < 128, iterations=kernel_size // 2)
  ```
- Line 1027: .get() with default: False
  ```python
  if context.get("has_high_frequency", False):
  ```
- Line 1029: .get() with default: False
  ```python
  elif context.get("is_smooth", False):
  ```

### strategies/tiled_expansion.py
Found 2 issues:

- Line 253: direct_assignment: y
  ```python
  y = 0
  ```
- Line 258: direct_assignment: x
  ```python
  
  ```

### utils/config_defaults.py
Found 52 issues:

- Line 26: Numeric constant: 0.8
  ```python
  "strength": 0.8,
  ```
- Line 27: Numeric constant: 50
  ```python
  "num_inference_steps": 50,
  ```
- Line 28: Numeric constant: 7.5
  ```python
  "guidance_scale": 7.5,
  ```
- Line 41: Numeric constant: 200
  ```python
  "default_high_threshold": 200,
  ```
- Line 43: Numeric constant: 3
  ```python
  "kernel_size": 3,
  ```
- Line 54: Numeric constant: 5
  ```python
  "default_radius": 5,
  ```
- Line 106: Numeric constant: 2000
  ```python
  "model_load": 2000,  # Per ControlNet model
  ```
- Line 107: Numeric constant: 1500
  ```python
  "operation_active": 1500,  # Additional for active operations
  ```
- Line 112: Numeric constant: 1000000
  ```python
  "megapixel_divisor": 1000000,  # 1e6 for MP calculations
  ```
- Line 128: Numeric constant: 6000
  ```python
  "generate": 6000,
  ```
- Line 129: Numeric constant: 5500
  ```python
  "inpaint": 5500,
  ```
- Line 130: Numeric constant: 5000
  ```python
  "img2img": 5000,
  ```
- Line 131: Numeric constant: 4000
  ```python
  "refine": 4000,
  ```
- Line 132: Numeric constant: 3500
  ```python
  "enhance": 3500,
  ```
- Line 133: Numeric constant: 8500
  ```python
  "controlnet_generate": 8500,
  ```
- Line 134: Numeric constant: 8000
  ```python
  "controlnet_inpaint": 8000,
  ```
- Line 135: Numeric constant: 7500
  ```python
  "controlnet_img2img": 7500,
  ```
- Line 139: Numeric constant: 8000
  ```python
  "generate": 8000,
  ```
- Line 140: Numeric constant: 7500
  ```python
  "inpaint": 7500,
  ```
- Line 141: Numeric constant: 7000
  ```python
  "img2img": 7000,
  ```
- Line 142: Numeric constant: 5000
  ```python
  "refine": 5000,
  ```
- Line 143: Numeric constant: 4500
  ```python
  "enhance": 4500,
  ```
- Line 144: Numeric constant: 10500
  ```python
  "controlnet_generate": 10500,
  ```
- Line 145: Numeric constant: 10000
  ```python
  "controlnet_inpaint": 10000,
  ```
- Line 146: Numeric constant: 9500
  ```python
  "controlnet_img2img": 9500,
  ```
- Line 150: Numeric constant: 12000
  ```python
  "generate": 12000,
  ```
- Line 151: Numeric constant: 11000
  ```python
  "inpaint": 11000,
  ```
- Line 152: Numeric constant: 10000
  ```python
  "img2img": 10000,
  ```
- Line 153: Numeric constant: 8000
  ```python
  "refine": 8000,
  ```
- Line 154: Numeric constant: 7000
  ```python
  "enhance": 7000,
  ```
- Line 155: Numeric constant: 14500
  ```python
  "controlnet_generate": 14500,
  ```
- Line 156: Numeric constant: 14000
  ```python
  "controlnet_inpaint": 14000,
  ```
- Line 157: Numeric constant: 13500
  ```python
  "controlnet_img2img": 13500,
  ```
- Line 161: Numeric constant: 3000
  ```python
  "generate": 3000,
  ```
- Line 162: Numeric constant: 2800
  ```python
  "inpaint": 2800,
  ```
- Line 163: Numeric constant: 2500
  ```python
  "img2img": 2500,
  ```
- Line 164: Numeric constant: 2000
  ```python
  "refine": 2000,
  ```
- Line 165: Numeric constant: 1800
  ```python
  "enhance": 1800,
  ```
- Line 166: Numeric constant: 4500
  ```python
  "controlnet_generate": 4500,
  ```
- Line 167: Numeric constant: 4300
  ```python
  "controlnet_inpaint": 4300,
  ```
- Line 168: Numeric constant: 4000
  ```python
  "controlnet_img2img": 4000,
  ```
- Line 172: Numeric constant: 4000
  ```python
  "generate": 4000,
  ```
- Line 173: Numeric constant: 3800
  ```python
  "inpaint": 3800,
  ```
- Line 174: Numeric constant: 3500
  ```python
  "img2img": 3500,
  ```
- Line 175: Numeric constant: 2500
  ```python
  "refine": 2500,
  ```
- Line 176: Numeric constant: 2300
  ```python
  "enhance": 2300,
  ```
- Line 177: Numeric constant: 5500
  ```python
  "controlnet_generate": 5500,
  ```
- Line 178: Numeric constant: 5300
  ```python
  "controlnet_inpaint": 5300,
  ```
- Line 179: Numeric constant: 5000
  ```python
  "controlnet_img2img": 5000,
  ```
- Line 184: Numeric constant: 150
  ```python
  "per_megapixel": 150,  # Additional MB per megapixel
  ```
- Line 185: Numeric constant: 0.8
  ```python
  "batch_size_multiplier": 0.8,  # Additional factor per batch item
  ```
- Line 189: Numeric constant: 200
  ```python
  "lora_overhead": 200,  # MB per LoRA
  ```

### utils/config_loader.py
Found 17 issues:

- Line 155: field_default: user_config
  ```python
  user_config: bool = False) -> Path:
  ```
- Line 300: Numeric constant: 80
  ```python
  "inference_steps": 80,
  ```
- Line 301: Numeric constant: 7.5
  ```python
  "cfg_scale": 7.5,
  ```
- Line 302: Numeric constant: 0.95
  ```python
  "denoise_strength": 0.95,
  ```
- Line 303: Numeric constant: 300
  ```python
  "blur_radius": 300,
  ```
- Line 306: Numeric constant: 60
  ```python
  "inference_steps": 60,
  ```
- Line 307: Numeric constant: 7.0
  ```python
  "cfg_scale": 7.0,
  ```
- Line 308: Numeric constant: 0.9
  ```python
  "denoise_strength": 0.9,
  ```
- Line 309: Numeric constant: 200
  ```python
  "blur_radius": 200,
  ```
- Line 312: Numeric constant: 40
  ```python
  "inference_steps": 40,
  ```
- Line 313: Numeric constant: 6.5
  ```python
  "cfg_scale": 6.5,
  ```
- Line 314: Numeric constant: 0.85
  ```python
  "denoise_strength": 0.85,
  ```
- Line 315: Numeric constant: 150
  ```python
  "blur_radius": 150,
  ```
- Line 318: Numeric constant: 25
  ```python
  "inference_steps": 25,
  ```
- Line 319: Numeric constant: 6.0
  ```python
  "cfg_scale": 6.0,
  ```
- Line 320: Numeric constant: 0.8
  ```python
  "denoise_strength": 0.8,
  ```
- Line 350: get_with_default: []
  ```python
  required_keys = schema.get('required', [])
  ```

### utils/config_migrator.py
Found 7 issues:

- Line 215: Numeric constant: 95
  ```python
  "output.formats.jpeg.quality": 95,
  ```
- Line 216: Numeric constant: 4.5
  ```python
  "vram.estimation.latent_multiplier": 4.5,
  ```
- Line 303: .get() with default: 'unknown'
  ```python
  header = f"""# Expandor Configuration v{config.get('version', 'unknown')}
  ```
- Line 313: Numeric constant: 120
  ```python
  yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)
  ```
- Line 315: field_default: dry_run
  ```python
  def migrate(self, dry_run: bool = False) -> bool:
  ```
- Line 318: Numeric constant: 80
  ```python
  print("=" * 80)
  ```
- Line 318: math_operations: 80
  ```python
  print("=" * 80)
  ```

### utils/dimension_calculator.py
Found 104 issues:

- Line 43: Numeric constant: 1920
  ```python
  "1080p": (1920, 1080),
  ```
- Line 43: Numeric constant: 1080
  ```python
  "1080p": (1920, 1080),
  ```
- Line 44: Numeric constant: 2560
  ```python
  "1440p": (2560, 1440),
  ```
- Line 44: Numeric constant: 1440
  ```python
  "1440p": (2560, 1440),
  ```
- Line 45: Numeric constant: 3840
  ```python
  "4K": (3840, 2160),
  ```
- Line 45: Numeric constant: 2160
  ```python
  "4K": (3840, 2160),
  ```
- Line 46: Numeric constant: 5120
  ```python
  "5K": (5120, 2880),
  ```
- Line 46: Numeric constant: 2880
  ```python
  "5K": (5120, 2880),
  ```
- Line 47: Numeric constant: 7680
  ```python
  "8K": (7680, 4320),
  ```
- Line 47: Numeric constant: 4320
  ```python
  "8K": (7680, 4320),
  ```
- Line 48: Numeric constant: 3440
  ```python
  "ultrawide_1440p": (3440, 1440),
  ```
- Line 48: Numeric constant: 1440
  ```python
  "ultrawide_1440p": (3440, 1440),
  ```
- Line 49: Numeric constant: 5120
  ```python
  "ultrawide_4K": (5120, 2160),
  ```
- Line 49: Numeric constant: 2160
  ```python
  "ultrawide_4K": (5120, 2160),
  ```
- Line 50: Numeric constant: 5760
  ```python
  "super_ultrawide": (5760, 1080),
  ```
- Line 50: Numeric constant: 1080
  ```python
  "super_ultrawide": (5760, 1080),
  ```
- Line 51: Numeric constant: 2160
  ```python
  "portrait_4K": (2160, 3840),
  ```
- Line 51: Numeric constant: 3840
  ```python
  "portrait_4K": (2160, 3840),
  ```
- Line 52: Numeric constant: 2880
  ```python
  "square_4K": (2880, 2880),
  ```
- Line 52: Numeric constant: 2880
  ```python
  "square_4K": (2880, 2880),
  ```
- Line 57: Numeric constant: 1024
  ```python
  (1024, 1024),  # 1:1
  ```
- Line 57: Numeric constant: 1024
  ```python
  (1024, 1024),  # 1:1
  ```
- Line 58: Numeric constant: 1152
  ```python
  (1152, 896),  # 4:3.11
  ```
- Line 58: Numeric constant: 896
  ```python
  (1152, 896),  # 4:3.11
  ```
- Line 59: Numeric constant: 1216
  ```python
  (1216, 832),  # 3:2.05
  ```
- Line 59: Numeric constant: 832
  ```python
  (1216, 832),  # 3:2.05
  ```
- Line 60: Numeric constant: 1344
  ```python
  (1344, 768),  # 16:9.14
  ```
- Line 60: Numeric constant: 768
  ```python
  (1344, 768),  # 16:9.14
  ```
- Line 61: Numeric constant: 1536
  ```python
  (1536, 640),  # 2.4:1
  ```
- Line 61: Numeric constant: 640
  ```python
  (1536, 640),  # 2.4:1
  ```
- Line 62: Numeric constant: 768
  ```python
  (768, 1344),  # 9:16 (portrait)
  ```
- Line 62: Numeric constant: 1344
  ```python
  (768, 1344),  # 9:16 (portrait)
  ```
- Line 63: Numeric constant: 896
  ```python
  (896, 1152),  # 3:4 (portrait)
  ```
- Line 63: Numeric constant: 1152
  ```python
  (896, 1152),  # 3:4 (portrait)
  ```
- Line 64: Numeric constant: 640
  ```python
  (640, 1536),  # 1:2.4 (tall portrait)
  ```
- Line 64: Numeric constant: 1536
  ```python
  (640, 1536),  # 1:2.4 (tall portrait)
  ```
- Line 69: Numeric constant: 16
  ```python
  "divisible_by": 16,
  ```
- Line 70: Numeric constant: 2048
  ```python
  "max_dimension": 2048,
  ```
- Line 71: Numeric constant: 1024
  ```python
  "optimal_pixels": 1024 * 1024,  # 1MP for best quality
  ```
- Line 71: Numeric constant: 1024
  ```python
  "optimal_pixels": 1024 * 1024,  # 1MP for best quality
  ```
- Line 71: math_operations: 1024
  ```python
  "optimal_pixels": 1024 * 1024,  # 1MP for best quality
  ```
- Line 93: .get() with default: 8
  ```python
  multiple = proc_config.get(
  ```
- Line 93: get_with_default: {}
  ```python
  multiple = proc_config.get(
  ```
- Line 105: math_operations: 93
  ```python
  Copy implementation from lines 70-93
  ```
- Line 115: Numeric constant: 1024
  ```python
  return (1024, 1024)
  ```
- Line 115: Numeric constant: 1024
  ```python
  return (1024, 1024)
  ```
- Line 122: math_operations: 118
  ```python
  """Copy from lines 95-118"""
  ```
- Line 146: math_operations: 142
  ```python
  """Copy from lines 120-142"""
  ```
- Line 157: Numeric constant: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 157: Numeric constant: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 157: math_operations: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 157: math_operations: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 158: Numeric constant: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 158: Numeric constant: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 158: math_operations: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 158: math_operations: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 166: Numeric constant: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 166: Numeric constant: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 166: math_operations: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 166: math_operations: 16
  ```python
  width = (width // 16) * 16
  ```
- Line 167: Numeric constant: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 167: Numeric constant: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 167: math_operations: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 167: math_operations: 16
  ```python
  height = (height // 16) * 16
  ```
- Line 171: function_default: 2.0
  ```python
  def calculate_progressive_strategy(
  ```
- Line 175: Numeric constant: 2.0
  ```python
  max_expansion_per_step: float = 2.0,
  ```
- Line 175: field_default: max_expansion_per_step
  ```python
  max_expansion_per_step: float = 2.0,
  ```
- Line 179: math_operations: 405
  ```python
  Copy core logic from lines 223-405 of resolution_manager.py
  ```
- Line 196: Numeric constant: 0.05
  ```python
  if abs(current_aspect - target_aspect) < 0.05:
  ```
- Line 196: if_comparison: 0.05
  ```python
  if abs(current_aspect - target_aspect) < 0.05:
  ```
- Line 203: Numeric constant: 8.0
  ```python
  if aspect_change_ratio > 8.0:
  ```
- Line 225: Numeric constant: 2.0
  ```python
  if total_expansion >= 2.0:
  ```
- Line 225: if_comparison: 2.0
  ```python
  if total_expansion >= 2.0:
  ```
- Line 226: Numeric constant: 2.0
  ```python
  next_w = min(int(temp_w * 2.0), target_w)
  ```
- Line 226: math_operations: 2.0
  ```python
  next_w = min(int(temp_w * 2.0), target_w)
  ```
- Line 245: direct_assignment: step_num
  ```python
  step_num = 2
  ```
- Line 246: Numeric constant: 0.95
  ```python
  while temp_w < target_w * 0.95:
  ```
- Line 246: math_operations: 0.95
  ```python
  while temp_w < target_w * 0.95:
  ```
- Line 247: Numeric constant: 1.5
  ```python
  if temp_w * 1.5 <= target_w:
  ```
- Line 247: math_operations: 1.5
  ```python
  if temp_w * 1.5 <= target_w:
  ```
- Line 248: Numeric constant: 1.5
  ```python
  next_w = int(temp_w * 1.5)
  ```
- Line 248: math_operations: 1.5
  ```python
  next_w = int(temp_w * 1.5)
  ```
- Line 270: Numeric constant: 10
  ```python
  if step_num > 10:
  ```
- Line 270: if_comparison: 10
  ```python
  if step_num > 10:
  ```
- Line 306: Numeric constant: 2.0
  ```python
  if total_expansion >= 2.0:
  ```
- Line 306: if_comparison: 2.0
  ```python
  if total_expansion >= 2.0:
  ```
- Line 307: Numeric constant: 2.0
  ```python
  next_h = min(int(temp_h * 2.0), target_h)
  ```
- Line 307: math_operations: 2.0
  ```python
  next_h = min(int(temp_h * 2.0), target_h)
  ```
- Line 326: direct_assignment: step_num
  ```python
  step_num = 2
  ```
- Line 327: Numeric constant: 0.95
  ```python
  while temp_h < target_h * 0.95:
  ```
- Line 327: math_operations: 0.95
  ```python
  while temp_h < target_h * 0.95:
  ```
- Line 328: Numeric constant: 1.5
  ```python
  if temp_h * 1.5 <= target_h:
  ```
- Line 328: math_operations: 1.5
  ```python
  if temp_h * 1.5 <= target_h:
  ```
- Line 329: Numeric constant: 1.5
  ```python
  next_h = int(temp_h * 1.5)
  ```
- Line 329: math_operations: 1.5
  ```python
  next_h = int(temp_h * 1.5)
  ```
- Line 351: Numeric constant: 10
  ```python
  if step_num > 10:
  ```
- Line 351: if_comparison: 10
  ```python
  if step_num > 10:
  ```
- Line 375: function_default: 0.8
  ```python
  def calculate_sliding_window_strategy(
  ```
- Line 379: Numeric constant: 200
  ```python
  window_size: int = 200,
  ```
- Line 379: field_default: window_size
  ```python
  window_size: int = 200,
  ```
- Line 380: Numeric constant: 0.8
  ```python
  overlap_ratio: float = 0.8,
  ```
- Line 380: field_default: overlap_ratio
  ```python
  overlap_ratio: float = 0.8,
  ```
- Line 409: direct_assignment: window_num
  ```python
  window_num = 1
  ```
- Line 456: direct_assignment: window_num
  ```python
  window_num = 1
  ```

### utils/image_utils.py
Found 25 issues:

- Line 25: field_default: fade_start
  ```python
  fade_start: float = 0.0) -> Image.Image:
  ```
- Line 84: function_default: "normal"
  ```python
  def blend_images(
  ```
- Line 88: field_default: mode
  ```python
  mode: str = "normal") -> Image.Image:
  ```
- Line 122: Numeric constant: 255.0
  ```python
  arr1 = np.array(img1, dtype=np.float32) / 255.0
  ```
- Line 122: math_operations: 255.0
  ```python
  arr1 = np.array(img1, dtype=np.float32) / 255.0
  ```
- Line 123: Numeric constant: 255.0
  ```python
  arr2 = np.array(img2, dtype=np.float32) / 255.0
  ```
- Line 123: math_operations: 255.0
  ```python
  arr2 = np.array(img2, dtype=np.float32) / 255.0
  ```
- Line 124: Numeric constant: 255.0
  ```python
  mask_arr = np.array(mask, dtype=np.float32) / 255.0
  ```
- Line 124: math_operations: 255.0
  ```python
  mask_arr = np.array(mask, dtype=np.float32) / 255.0
  ```
- Line 145: function_default: "average"
  ```python
  def extract_edge_colors(
  ```
- Line 146: field_default: width
  ```python
  image: Image.Image, edge: str, width: int = 1, method: str = "average"
  ```
- Line 146: field_default: method
  ```python
  image: Image.Image, edge: str, width: int = 1, method: str = "average"
  ```
- Line 190: Numeric constant: 3
  ```python
  result = np.zeros((edge_pixels.shape[0], 3))
  ```
- Line 192: Numeric constant: 3
  ```python
  for c in range(3):
  ```
- Line 192: range_calls: 3
  ```python
  for c in range(3):
  ```
- Line 198: Numeric constant: 3
  ```python
  result = np.zeros((edge_pixels.shape[1], 3))
  ```
- Line 200: Numeric constant: 3
  ```python
  for c in range(3):
  ```
- Line 200: range_calls: 3
  ```python
  for c in range(3):
  ```
- Line 273: Numeric constant: 1e-08
  ```python
  noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
  ```
- Line 278: field_default: feather
  ```python
  size: Tuple[int, int], center: Tuple[int, int], radius: int, feather: int = 0
  ```
- Line 323: math_operations: 4
  ```python
  sigma: Standard deviation (default: size/4)
  ```
- Line 329: Numeric constant: 4.0
  ```python
  sigma = size / 4.0
  ```
- Line 329: math_operations: 4.0
  ```python
  sigma = size / 4.0
  ```
- Line 332: Numeric constant: 2.0
  ```python
  center = size / 2.0
  ```
- Line 332: math_operations: 2.0
  ```python
  center = size / 2.0
  ```

### utils/installation_validator.py
Found 38 issues:

- Line 33: field_default: verbose
  ```python
  def validate_installation(self, verbose: bool = True) -> Dict[str, Any]:
  ```
- Line 79: Numeric constant: 3
  ```python
  min_version = (3, 8)
  ```
- Line 113: Numeric constant: 1024
  ```python
  i).total_memory / (1024**3)  # GB
  ```
- Line 113: Numeric constant: 3
  ```python
  i).total_memory / (1024**3)  # GB
  ```
- Line 125: Numeric constant: 1024
  ```python
  total_vram_gb = total_vram / (1024**3)
  ```
- Line 125: Numeric constant: 3
  ```python
  total_vram_gb = total_vram / (1024**3)
  ```
- Line 125: math_operations: 3
  ```python
  total_vram_gb = total_vram / (1024**3)
  ```
- Line 312: get_with_default: package_name
  ```python
  module_name = config.get("module", package_name)
  ```
- Line 339: Numeric constant: 1024
  ```python
  free_gb = disk_usage.free / (1024**3)
  ```
- Line 339: Numeric constant: 3
  ```python
  free_gb = disk_usage.free / (1024**3)
  ```
- Line 339: math_operations: 3
  ```python
  free_gb = disk_usage.free / (1024**3)
  ```
- Line 341: Numeric constant: 50
  ```python
  if free_gb < 50:
  ```
- Line 341: if_comparison: 50
  ```python
  if free_gb < 50:
  ```
- Line 359: Numeric constant: 1024
  ```python
  total_ram_gb = memory.total / (1024**3)
  ```
- Line 359: Numeric constant: 3
  ```python
  total_ram_gb = memory.total / (1024**3)
  ```
- Line 359: math_operations: 3
  ```python
  total_ram_gb = memory.total / (1024**3)
  ```
- Line 361: Numeric constant: 16
  ```python
  if total_ram_gb < 16:
  ```
- Line 361: if_comparison: 16
  ```python
  if total_ram_gb < 16:
  ```
- Line 385: Numeric constant: 4
  ```python
  if strategy_count < 4:
  ```
- Line 481: Numeric constant: 60
  ```python
  print("\n" + "=" * 60)
  ```
- Line 481: math_operations: 60
  ```python
  print("\n" + "=" * 60)
  ```
- Line 483: Numeric constant: 60
  ```python
  print("=" * 60)
  ```
- Line 483: math_operations: 60
  ```python
  print("=" * 60)
  ```
- Line 490: .get() with default: 'unknown'
  ```python
  results.get(
  ```
- Line 490: get_with_default: {}
  ```python
  results.get(
  ```
- Line 492: get_with_default: 'unknown'
  ```python
  {}).get(
  ```
- Line 501: get_with_default: {}
  ```python
  info = results.get("info", {})
  ```
- Line 502: .get() with default: 'unknown'
  ```python
  print(f"  • CUDA Version: {info.get('cuda_version', 'unknown')}")
  ```
- Line 502: get_with_default: 'unknown'
  ```python
  print(f"  • CUDA Version: {info.get('cuda_version', 'unknown')}")
  ```
- Line 503: .get() with default: 'unknown'
  ```python
  print(f"  • Total VRAM: {info.get('total_vram_gb', 'unknown')}GB")
  ```
- Line 503: get_with_default: 'unknown'
  ```python
  print(f"  • Total VRAM: {info.get('total_vram_gb', 'unknown')}GB")
  ```
- Line 505: get_with_default: []
  ```python
  for gpu in info.get("gpus", []):
  ```
- Line 533: get_with_default: {}
  ```python
  features = results.get("info", {}).get("available_features", [])
  ```
- Line 533: get_with_default: []
  ```python
  features = results.get("info", {}).get("available_features", [])
  ```
- Line 540: Numeric constant: 60
  ```python
  print("\n" + "=" * 60)
  ```
- Line 540: math_operations: 60
  ```python
  print("\n" + "=" * 60)
  ```
- Line 545: Numeric constant: 60
  ```python
  print("=" * 60 + "\n")
  ```
- Line 545: math_operations: 60
  ```python
  print("=" * 60 + "\n")
  ```

### utils/logging_utils.py
Found 1 issues:

- Line 24: get_with_default: self.RESET
  ```python
  log_color = self.COLORS.get(record.levelname, self.RESET)
  ```

### utils/memory_utils.py
Found 45 issues:

- Line 53: Numeric constant: 1024
  ```python
  gpu_free = gpu_stats[0] / (1024**2)  # Convert to MB
  ```
- Line 54: Numeric constant: 1024
  ```python
  gpu_total = gpu_stats[1] / (1024**2)
  ```
- Line 61: Numeric constant: 1024
  ```python
  cpu_total = cpu_info.total / (1024**2)
  ```
- Line 62: Numeric constant: 1024
  ```python
  cpu_free = cpu_info.available / (1024**2)
  ```
- Line 63: Numeric constant: 1024
  ```python
  cpu_used = cpu_info.used / (1024**2)
  ```
- Line 85: field_default: aggressive
  ```python
  def clear_cache(self, aggressive: bool = False):
  ```
- Line 116: direct_assignment: num_elements
  ```python
  num_elements = 1
  ```
- Line 122: Numeric constant: 4
  ```python
  torch.float32: 4,
  ```
- Line 125: Numeric constant: 4
  ```python
  torch.int32: 4,
  ```
- Line 142: Numeric constant: 1024
  ```python
  return total_bytes / (1024**2)
  ```
- Line 144: function_default: 1.2
  ```python
  def has_sufficient_memory(
  ```
- Line 145: Numeric constant: 1.2
  ```python
  self, required_mb: float, safety_factor: float = 1.2
  ```
- Line 145: field_default: safety_factor
  ```python
  self, required_mb: float, safety_factor: float = 1.2
  ```
- Line 163: field_default: name
  ```python
  self, name: str = "operation", clear_on_exit: bool = True
  ```
- Line 163: field_default: clear_on_exit
  ```python
  self, name: str = "operation", clear_on_exit: bool = True
  ```
- Line 186: function_default: 0.8
  ```python
  def get_optimal_batch_size(
  ```
- Line 190: Numeric constant: 16
  ```python
  max_batch_size: int = 16,
  ```
- Line 190: field_default: max_batch_size
  ```python
  max_batch_size: int = 16,
  ```
- Line 191: Numeric constant: 0.8
  ```python
  safety_factor: float = 0.8,
  ```
- Line 191: field_default: safety_factor
  ```python
  safety_factor: float = 0.8,
  ```
- Line 239: math_operations: 1024
  ```python
  tensor.nelement() /
  ```
- Line 240: Numeric constant: 1024
  ```python
  1024**2:.1f}MB)"
  ```
- Line 271: math_operations: 1024
  ```python
  tensor.nelement() /
  ```
- Line 272: Numeric constant: 1024
  ```python
  1024**2:.1f}MB)"
  ```
- Line 283: field_default: include_gradients
  ```python
  def estimate_model_memory(model: Any, include_gradients: bool = True) -> float:
  ```
- Line 294: direct_assignment: total_params
  ```python
  total_params = 0
  ```
- Line 295: direct_assignment: total_bytes
  ```python
  total_bytes = 0
  ```
- Line 312: .get() with default: 2
  ```python
  gradient_multiplier = proc_config.get(
  ```
- Line 312: get_with_default: {}
  ```python
  gradient_multiplier = proc_config.get(
  ```
- Line 331: get_with_default: {}
  ```python
  mem_params = proc_config.get('memory_params', {})
  ```
- Line 333: .get() with default: 4
  ```python
  mem_params.get(
  ```
- Line 333: get_with_default: 4
  ```python
  mem_params.get(
  ```
- Line 335: Numeric constant: 4
  ```python
  4) if include_gradients else mem_params.get(
  ```
- Line 335: .get() with default: 2
  ```python
  4) if include_gradients else mem_params.get(
  ```
- Line 343: Numeric constant: 1024
  ```python
  total_mb = total_bytes / (1024**2)
  ```
- Line 357: function_default: 2048
  ```python
  def calculate_optimal_tile_size(
  ```
- Line 360: Numeric constant: 12
  ```python
  bytes_per_pixel: float = 12,
  ```
- Line 360: field_default: bytes_per_pixel
  ```python
  bytes_per_pixel: float = 12,
  ```
- Line 361: Numeric constant: 64
  ```python
  overlap: int = 64,
  ```
- Line 361: field_default: overlap
  ```python
  overlap: int = 64,
  ```
- Line 362: Numeric constant: 256
  ```python
  min_tile: int = 256,
  ```
- Line 362: field_default: min_tile
  ```python
  min_tile: int = 256,
  ```
- Line 363: Numeric constant: 2048
  ```python
  max_tile: int = 2048,
  ```
- Line 363: field_default: max_tile
  ```python
  max_tile: int = 2048,
  ```
- Line 388: Numeric constant: 1024
  ```python
  tile_memory_mb = (tile_pixels * bytes_per_pixel) / (1024**2)
  ```

### utils/model_manager.py
Found 43 issues:

- Line 65: or_pattern: "
                "
  ```python
  "Please add 'paths.cache_dir' to your configuration file or "
  ```
- Line 75: field_default: force_download
  ```python
  force_download: bool = False,
  ```
- Line 340: Numeric constant: 1024
  ```python
  if size_bytes < 1024 * 1024:
  ```
- Line 340: Numeric constant: 1024
  ```python
  if size_bytes < 1024 * 1024:
  ```
- Line 340: math_operations: 1024
  ```python
  if size_bytes < 1024 * 1024:
  ```
- Line 341: Numeric constant: 1024
  ```python
  return f"{size_bytes / 1024:.1f} KB"
  ```
- Line 341: math_operations: 1024
  ```python
  return f"{size_bytes / 1024:.1f} KB"
  ```
- Line 342: Numeric constant: 1024
  ```python
  elif size_bytes < 1024 * 1024 * 1024:
  ```
- Line 342: Numeric constant: 1024
  ```python
  elif size_bytes < 1024 * 1024 * 1024:
  ```
- Line 342: Numeric constant: 1024
  ```python
  elif size_bytes < 1024 * 1024 * 1024:
  ```
- Line 342: math_operations: 1024
  ```python
  elif size_bytes < 1024 * 1024 * 1024:
  ```
- Line 342: math_operations: 1024
  ```python
  elif size_bytes < 1024 * 1024 * 1024:
  ```
- Line 343: Numeric constant: 1024
  ```python
  return f"{size_bytes / (1024 * 1024):.1f} MB"
  ```
- Line 343: Numeric constant: 1024
  ```python
  return f"{size_bytes / (1024 * 1024):.1f} MB"
  ```
- Line 343: math_operations: 1024
  ```python
  return f"{size_bytes / (1024 * 1024):.1f} MB"
  ```
- Line 345: Numeric constant: 1024
  ```python
  return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
  ```
- Line 345: Numeric constant: 1024
  ```python
  return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
  ```
- Line 345: Numeric constant: 1024
  ```python
  return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
  ```
- Line 345: math_operations: 1024
  ```python
  return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
  ```
- Line 345: math_operations: 1024
  ```python
  return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
  ```
- Line 436: Numeric constant: 0.25
  ```python
  return int(base_size * 0.25)
  ```
- Line 436: math_operations: 0.25
  ```python
  return int(base_size * 0.25)
  ```
- Line 442: Numeric constant: 7
  ```python
  return 7 * 1024 * 1024 * 1024  # 7GB for SDXL models
  ```
- Line 442: Numeric constant: 1024
  ```python
  return 7 * 1024 * 1024 * 1024  # 7GB for SDXL models
  ```
- Line 442: Numeric constant: 1024
  ```python
  return 7 * 1024 * 1024 * 1024  # 7GB for SDXL models
  ```
- Line 442: Numeric constant: 1024
  ```python
  return 7 * 1024 * 1024 * 1024  # 7GB for SDXL models
  ```
- Line 442: math_operations: 1024
  ```python
  return 7 * 1024 * 1024 * 1024  # 7GB for SDXL models
  ```
- Line 444: Numeric constant: 15
  ```python
  return 15 * 1024 * 1024 * 1024  # 15GB for FLUX models
  ```
- Line 444: Numeric constant: 1024
  ```python
  return 15 * 1024 * 1024 * 1024  # 15GB for FLUX models
  ```
- Line 444: Numeric constant: 1024
  ```python
  return 15 * 1024 * 1024 * 1024  # 15GB for FLUX models
  ```
- Line 444: Numeric constant: 1024
  ```python
  return 15 * 1024 * 1024 * 1024  # 15GB for FLUX models
  ```
- Line 444: math_operations: 1024
  ```python
  return 15 * 1024 * 1024 * 1024  # 15GB for FLUX models
  ```
- Line 446: Numeric constant: 5
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 446: Numeric constant: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 446: Numeric constant: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 446: Numeric constant: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 446: math_operations: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 452: Numeric constant: 5
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 452: Numeric constant: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 452: Numeric constant: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 452: Numeric constant: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 452: math_operations: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```
- Line 452: math_operations: 1024
  ```python
  return 5 * 1024 * 1024 * 1024  # 5GB default
  ```

### utils/path_resolver.py
Found 5 issues:

- Line 16: function_default: "directory"
  ```python
  def resolve_path(self, path_config: Union[str, Path],
  ```
- Line 17: field_default: create
  ```python
  create: bool = True,
  ```
- Line 18: field_default: path_type
  ```python
  path_type: str = "directory") -> Path:
  ```
- Line 94: function_default: "data"
  ```python
  def get_writable_dir(self, preferred_paths: list,
  ```
- Line 95: field_default: purpose
  ```python
  purpose: str = "data") -> Path:
  ```

### utils/vram_manager.py
Found 6 issues:

- Line 50: function_default: "sdxl"
  ```python
  def estimate_operation_vram(
  ```
- Line 51: field_default: model_type
  ```python
  self, operation: str, width: int, height: int, model_type: str = "sdxl"
  ```
- Line 81: Numeric constant: 1024
  ```python
  pixel_overhead = (pixels / (1024 * 1024)) * pixel_overhead_per_megapixel
  ```
- Line 81: Numeric constant: 1024
  ```python
  pixel_overhead = (pixels / (1024 * 1024)) * pixel_overhead_per_megapixel
  ```
- Line 81: math_operations: 1024
  ```python
  pixel_overhead = (pixels / (1024 * 1024)) * pixel_overhead_per_megapixel
  ```
- Line 149: Numeric constant: 1.1
  ```python
  return available >= required_mb * 1.1  # 10% safety margin
  ```


## Priority Fixes

1. Fix all `.get()` calls with defaults - use ConfigurationManager
2. Remove all `or` patterns with fallback values
3. Move all numeric constants to configuration files
4. Replace direct assignments with config lookups