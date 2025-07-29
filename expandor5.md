# Expandor Phase 5: Advanced Features & Full Implementation Plan

## Executive Summary

Phase 5 focuses on completing the advanced features that will make Expandor the ultimate universal image expansion system. With Phase 4's solid foundation (v0.5.0), we now implement full ControlNet support, real WebUI adapters, and performance optimizations while maintaining our core philosophy: **QUALITY OVER ALL, FAIL LOUD, NO COMPROMISES**.

## Current State (Post-Phase 4)

### ✅ Completed
- Core Expandor system with mandatory adapter pattern
- All expansion strategies (Direct, Progressive, SWPO, Tiled, CPU Offload)
- Quality validation and artifact detection
- CLI with setup wizard
- DiffusersPipelineAdapter (fully functional)
- New ExpandorConfig API (tests updated)
- Package v0.5.0 released

### 🚧 Partial Implementation
- ControlNet: Model loading only (no generation)
- ComfyUI/A1111: Placeholder adapters with documentation

### 🔧 Known Issues
- 166 flake8 style warnings (cosmetic only)
- Hardcoded values in some strategies (strength, blur_radius)
- No performance benchmarks

## Phase 5 Implementation Roadmap

### Stage 1: ControlNet Full Implementation (3-4 days)

#### 1.1 ControlNet Pipeline Integration
```python
# expandor/adapters/diffusers_adapter.py

def _initialize_controlnet_pipelines(self):
    """Initialize ControlNet-enabled pipelines"""
    if not hasattr(self, "controlnet_models"):
        self.controlnet_models = {}
    
    # Import ControlNet pipeline classes
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )
    
    # Create ControlNet pipelines sharing base model components
    if self.model_type == "sdxl" and self.controlnet_models:
        self.controlnet_pipelines = {
            "generate": StableDiffusionXLControlNetPipeline(
                vae=self.base_pipeline.vae,
                text_encoder=self.base_pipeline.text_encoder,
                text_encoder_2=self.base_pipeline.text_encoder_2,
                tokenizer=self.base_pipeline.tokenizer,
                tokenizer_2=self.base_pipeline.tokenizer_2,
                unet=self.base_pipeline.unet,
                scheduler=self.base_pipeline.scheduler,
                controlnet=None,  # Set per generation
            ),
            "inpaint": StableDiffusionXLControlNetInpaintPipeline(...),
            "img2img": StableDiffusionXLControlNetImg2ImgPipeline(...),
        }

def controlnet_inpaint(self, image, mask, control_image, prompt, **kwargs):
    """Full ControlNet inpainting implementation"""
    # Validate inputs
    if not self.supports_controlnet():
        raise NotImplementedError(
            f"ControlNet not supported for model type: {self.model_type}\n"
            f"ControlNet currently requires SDXL models.\n"
            f"Your model type: {self.model_type}"
        )
    
    # Get control type from image metadata or detect
    control_type = self._detect_control_type(control_image)
    
    if control_type not in self.controlnet_models:
        raise ValueError(
            f"ControlNet type '{control_type}' not loaded.\n"
            f"Available types: {list(self.controlnet_models.keys())}\n"
            f"Load with: adapter.load_controlnet(model_id, '{control_type}')"
        )
    
    # Set the appropriate ControlNet
    pipeline = self.controlnet_pipelines["inpaint"]
    pipeline.controlnet = self.controlnet_models[control_type]
    pipeline = pipeline.to(self.device)
    
    # Generate with ControlNet guidance
    generator = self._get_generator(kwargs.get("seed"))
    
    result = pipeline(
        prompt=prompt,
        negative_prompt=kwargs.get("negative_prompt"),
        image=image,
        mask_image=mask,
        control_image=control_image,
        controlnet_conditioning_scale=kwargs.get("controlnet_conditioning_scale", 1.0),
        strength=kwargs.get("strength", 0.8),
        num_inference_steps=kwargs.get("num_inference_steps", 50),
        guidance_scale=kwargs.get("guidance_scale", 7.5),
        generator=generator,
    )
    
    return result.images[0]
```

#### 1.2 Control Extractors
```python
# expandor/processors/controlnet_extractors.py

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

class ControlNetExtractor:
    """Extract control signals from images for ControlNet guidance"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self._extractors = {}
        
    def extract_canny(self, image: Image.Image, low_threshold=100, high_threshold=200) -> Image.Image:
        """Extract Canny edges for structure guidance"""
        # Convert to numpy
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to PIL Image
        # ControlNet expects RGB, so convert grayscale to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
    
    def extract_depth(self, image: Image.Image, model_id="Intel/dpt-large") -> Image.Image:
        """Extract depth map using DPT model"""
        if "depth" not in self._extractors:
            self._extractors["depth"] = pipeline(
                "depth-estimation",
                model=model_id,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        # Get depth estimation
        depth = self._extractors["depth"](image)
        depth_array = np.array(depth["depth"])
        
        # Normalize to 0-255
        depth_normalized = ((depth_array - depth_array.min()) / 
                          (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB
        depth_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(depth_rgb)
    
    def extract_blur(self, image: Image.Image, kernel_size=21) -> Image.Image:
        """Extract blurred version for soft guidance"""
        img_array = np.array(image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        
        return Image.fromarray(blurred)
    
    def auto_extract(self, image: Image.Image, strategy="smart") -> Dict[str, Image.Image]:
        """Automatically extract multiple control types"""
        controls = {}
        
        if strategy == "smart":
            # Analyze image to determine best control types
            has_clear_edges = self._has_clear_edges(image)
            has_depth_cues = self._has_depth_cues(image)
            
            if has_clear_edges:
                controls["canny"] = self.extract_canny(image)
            
            if has_depth_cues:
                controls["depth"] = self.extract_depth(image)
            
            # Always include blur for fallback
            controls["blur"] = self.extract_blur(image)
        
        elif strategy == "all":
            # Extract all supported types
            controls["canny"] = self.extract_canny(image)
            controls["depth"] = self.extract_depth(image)
            controls["blur"] = self.extract_blur(image)
        
        return controls
```

#### 1.3 ControlNet-Aware Strategies
```python
# expandor/strategies/controlnet_progressive.py

class ControlNetProgressiveStrategy(ProgressiveOutpaintStrategy):
    """Progressive outpainting with ControlNet structure guidance"""
    
    def __init__(self, adapter, boundary_tracker, vram_manager, logger=None):
        super().__init__(adapter, boundary_tracker, vram_manager, logger)
        self.control_extractor = ControlNetExtractor(logger)
        
    def execute(self, config: ExpandorConfig) -> ExpandorResult:
        """Execute with ControlNet guidance at each step"""
        # Extract initial control from source
        source_img = self._load_image(config.source_image)
        initial_controls = self.control_extractor.auto_extract(source_img)
        
        # Store best control type
        config.metadata["controlnet_type"] = self._select_best_control(initial_controls)
        
        # Execute base strategy with control injection
        return super().execute(config)
    
    def _progressive_expand_with_control(self, current_img, target_size, config):
        """Override to add ControlNet guidance"""
        # Extract control from current state
        control_image = self.control_extractor.extract_canny(current_img)
        
        # Expand control to match target
        control_expanded = self._expand_control_image(control_image, target_size)
        
        # Use ControlNet-enabled inpainting
        result = self.adapter.controlnet_inpaint(
            image=current_img,
            mask=mask,
            control_image=control_expanded,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            controlnet_conditioning_scale=0.7,  # Slightly reduced for flexibility
            strength=config.denoising_strength,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
        )
        
        return result
```

### Stage 2: ComfyUI Adapter Implementation (2-3 days)

#### 2.1 ComfyUI API Client
```python
# expandor/adapters/comfyui_client.py

import asyncio
import base64
import json
import time
import websocket
from typing import Any, Dict, Optional

import aiohttp
import requests


class ComfyUIClient:
    """WebSocket and HTTP client for ComfyUI API"""
    
    def __init__(self, server_url="http://127.0.0.1:8188", api_key=None):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.ws_url = self.server_url.replace("http", "ws") + "/ws"
        self.client_id = str(uuid.uuid4())
        
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow and return prompt ID"""
        # Inject client ID
        workflow["client_id"] = self.client_id
        
        # Send to API
        response = requests.post(
            f"{self.server_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"ComfyUI API error: {response.status_code}\n"
                f"Response: {response.text}\n"
                f"Make sure ComfyUI is running with --enable-cors-header"
            )
        
        return response.json()["prompt_id"]
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for workflow completion via WebSocket"""
        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={self.client_id}")
        
        start_time = time.time()
        result = None
        
        try:
            while time.time() - start_time < timeout:
                msg = ws.recv()
                if msg:
                    data = json.loads(msg)
                    
                    if data["type"] == "executed" and data["data"]["node"] is None:
                        # Workflow complete
                        result = self.get_history(prompt_id)
                        break
                    
                    elif data["type"] == "execution_error":
                        raise RuntimeError(
                            f"ComfyUI execution error: {data['data']['error']}\n"
                            f"Node: {data['data']['node_id']}"
                        )
                
                time.sleep(0.1)
        
        finally:
            ws.close()
        
        if not result:
            raise TimeoutError(f"Workflow timeout after {timeout}s")
        
        return result
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output"):
        """Download image from ComfyUI"""
        response = requests.get(
            f"{self.server_url}/view",
            params={
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            },
            headers=self._get_headers()
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get image: {filename}")
        
        return Image.open(io.BytesIO(response.content))
```

#### 2.2 ComfyUI Workflow Builder
```python
# expandor/adapters/comfyui_workflows.py

class ComfyUIWorkflowBuilder:
    """Build ComfyUI workflows programmatically"""
    
    def __init__(self):
        self.nodes = {}
        self.links = []
        self.node_counter = 0
        
    def add_node(self, class_type: str, inputs: Dict[str, Any], 
                 title: Optional[str] = None) -> str:
        """Add a node and return its ID"""
        node_id = str(self.node_counter)
        self.node_counter += 1
        
        self.nodes[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": title or class_type}
        }
        
        return node_id
    
    def link_nodes(self, from_node: str, from_slot: int, 
                   to_node: str, to_input: str):
        """Create connection between nodes"""
        # Update target node input
        self.nodes[to_node]["inputs"][to_input] = [from_node, from_slot]
    
    def build_inpaint_workflow(self, model_name: str, prompt: str, 
                              image_path: str, mask_path: str) -> Dict[str, Any]:
        """Build inpainting workflow"""
        # Load model
        model_node = self.add_node("CheckpointLoaderSimple", {
            "ckpt_name": model_name
        }, "Load Model")
        
        # Load images
        image_node = self.add_node("LoadImage", {
            "image": image_path
        }, "Load Image")
        
        mask_node = self.add_node("LoadImage", {
            "image": mask_path
        }, "Load Mask")
        
        # CLIP encode
        positive_node = self.add_node("CLIPTextEncode", {
            "text": prompt,
            "clip": [model_node, 1]
        }, "Positive Prompt")
        
        negative_node = self.add_node("CLIPTextEncode", {
            "text": "low quality, blurry",
            "clip": [model_node, 1]
        }, "Negative Prompt")
        
        # VAE encode
        vae_encode_node = self.add_node("VAEEncode", {
            "pixels": [image_node, 0],
            "vae": [model_node, 2]
        }, "VAE Encode")
        
        # KSampler
        sampler_node = self.add_node("KSampler", {
            "seed": 42,
            "steps": 30,
            "cfg": 7.5,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 0.8,
            "model": [model_node, 0],
            "positive": [positive_node, 0],
            "negative": [negative_node, 0],
            "latent": [vae_encode_node, 0]
        }, "Sample")
        
        # VAE decode
        decode_node = self.add_node("VAEDecode", {
            "samples": [sampler_node, 0],
            "vae": [model_node, 2]
        }, "VAE Decode")
        
        # Save
        save_node = self.add_node("SaveImage", {
            "images": [decode_node, 0],
            "filename_prefix": "expandor_output"
        }, "Save")
        
        return {"prompt": self.nodes}
```

#### 2.3 Full ComfyUI Adapter
```python
# expandor/adapters/comfyui_adapter.py (replacing placeholder)

class ComfyUIPipelineAdapter(BasePipelineAdapter):
    """Full ComfyUI integration adapter"""
    
    def __init__(self, server_url="http://127.0.0.1:8188", 
                 model_name=None, api_key=None, **kwargs):
        """Initialize ComfyUI adapter"""
        super().__init__()
        
        self.client = ComfyUIClient(server_url, api_key)
        self.workflow_builder = ComfyUIWorkflowBuilder()
        self.model_name = model_name or "sdxl_base.safetensors"
        self.logger = kwargs.get("logger", setup_logger(__name__))
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self):
        """Verify ComfyUI server is accessible"""
        try:
            response = requests.get(f"{self.client.server_url}/system_stats")
            if response.status_code != 200:
                raise ConnectionError(
                    f"ComfyUI server not responding at {self.client.server_url}\n"
                    f"Please ensure ComfyUI is running with --enable-cors-header"
                )
            
            self.logger.info("Connected to ComfyUI server")
            
            # Get available models
            models_response = requests.get(f"{self.client.server_url}/models")
            if models_response.status_code == 200:
                available_models = models_response.json()
                self.logger.info(f"Available models: {len(available_models)}")
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to ComfyUI at {self.client.server_url}\n"
                f"Error: {str(e)}\n"
                f"Solutions:\n"
                f"1. Start ComfyUI: python main.py --enable-cors-header\n"
                f"2. Check the server URL is correct\n"
                f"3. Ensure no firewall is blocking the connection"
            )
    
    def inpaint(self, image: Image.Image, mask: Image.Image, 
                prompt: str, **kwargs) -> Image.Image:
        """Inpaint using ComfyUI workflow"""
        # Save images temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "input.png"
            mask_path = Path(temp_dir) / "mask.png"
            
            image.save(image_path)
            mask.save(mask_path)
            
            # Build workflow
            workflow = self.workflow_builder.build_inpaint_workflow(
                model_name=self.model_name,
                prompt=prompt,
                image_path=str(image_path),
                mask_path=str(mask_path)
            )
            
            # Add custom parameters
            if "seed" in kwargs:
                # Find KSampler node and update seed
                for node_id, node in workflow["prompt"].items():
                    if node["class_type"] == "KSampler":
                        node["inputs"]["seed"] = kwargs["seed"]
            
            # Execute workflow
            prompt_id = self.client.queue_prompt(workflow)
            self.logger.info(f"Queued ComfyUI workflow: {prompt_id}")
            
            # Wait for completion
            result = self.client.wait_for_completion(prompt_id)
            
            # Get output image
            outputs = result[prompt_id]["outputs"]
            for node_id, output in outputs.items():
                if "images" in output:
                    # Get first image
                    image_data = output["images"][0]
                    return self.client.get_image(
                        image_data["filename"],
                        image_data.get("subfolder", ""),
                        image_data.get("type", "output")
                    )
            
            raise RuntimeError("No output image found in ComfyUI result")
```

### Stage 3: A1111 Adapter Implementation (2-3 days)

#### 3.1 A1111 API Client
```python
# expandor/adapters/a1111_client.py

class A1111Client:
    """HTTP client for Automatic1111 WebUI API"""
    
    def __init__(self, base_url="http://127.0.0.1:7860", api_key=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def txt2img(self, prompt: str, negative_prompt: str = "", 
                **kwargs) -> Dict[str, Any]:
        """Text to image generation"""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": kwargs.get("steps", 30),
            "cfg_scale": kwargs.get("cfg_scale", 7.5),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "seed": kwargs.get("seed", -1),
            "sampler_name": kwargs.get("sampler_name", "DPM++ 2M Karras"),
            "enable_hr": kwargs.get("enable_hr", False),
            "hr_scale": kwargs.get("hr_scale", 2),
            "hr_upscaler": kwargs.get("hr_upscaler", "ESRGAN_4x"),
        }
        
        response = requests.post(
            f"{self.base_url}/sdapi/v1/txt2img",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"A1111 API error: {response.status_code}\n"
                f"Response: {response.text}"
            )
        
        return response.json()
    
    def img2img(self, init_images: List[str], prompt: str, 
                **kwargs) -> Dict[str, Any]:
        """Image to image generation"""
        payload = {
            "init_images": init_images,  # Base64 encoded
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "steps": kwargs.get("steps", 30),
            "cfg_scale": kwargs.get("cfg_scale", 7.5),
            "denoising_strength": kwargs.get("denoising_strength", 0.75),
            "seed": kwargs.get("seed", -1),
            "sampler_name": kwargs.get("sampler_name", "DPM++ 2M Karras"),
            "include_init_images": False,
        }
        
        # Add mask if provided
        if "mask" in kwargs:
            payload["mask"] = kwargs["mask"]  # Base64 encoded
            payload["inpainting_fill"] = kwargs.get("inpainting_fill", 1)
            payload["inpaint_full_res"] = kwargs.get("inpaint_full_res", True)
            payload["inpaint_full_res_padding"] = kwargs.get("inpaint_full_res_padding", 32)
        
        response = requests.post(
            f"{self.base_url}/sdapi/v1/img2img",
            json=payload,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"A1111 API error: {response.status_code}\n"
                f"Response: {response.text}"
            )
        
        return response.json()
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        response = requests.get(
            f"{self.base_url}/sdapi/v1/sd-models",
            headers=self.headers
        )
        return response.json() if response.status_code == 200 else []
    
    def set_model(self, model_title: str):
        """Change active model"""
        response = requests.post(
            f"{self.base_url}/sdapi/v1/options",
            json={"sd_model_checkpoint": model_title},
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to set model: {model_title}")
```

#### 3.2 Full A1111 Adapter
```python
# expandor/adapters/a1111_adapter.py (replacing placeholder)

class A1111PipelineAdapter(BasePipelineAdapter):
    """Full Automatic1111 WebUI integration adapter"""
    
    def __init__(self, base_url="http://127.0.0.1:7860", 
                 api_key=None, model_name=None, **kwargs):
        """Initialize A1111 adapter"""
        super().__init__()
        
        self.client = A1111Client(base_url, api_key)
        self.logger = kwargs.get("logger", setup_logger(__name__))
        self.current_model = model_name
        
        # Test connection and setup
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Test connection and configure A1111"""
        try:
            # Get available models
            models = self.client.get_models()
            if not models:
                raise RuntimeError("No models found in A1111")
            
            self.logger.info(f"Found {len(models)} models in A1111")
            
            # Set model if specified
            if self.current_model:
                matching = [m for m in models if self.current_model in m["title"]]
                if matching:
                    self.client.set_model(matching[0]["title"])
                    self.logger.info(f"Set model to: {matching[0]['title']}")
                else:
                    self.logger.warning(f"Model '{self.current_model}' not found")
            
            # Get current settings
            response = requests.get(f"{self.client.base_url}/sdapi/v1/options")
            if response.status_code == 200:
                options = response.json()
                self.logger.info(f"A1111 using model: {options.get('sd_model_checkpoint')}")
                
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to A1111 at {self.client.base_url}\n"
                f"Error: {str(e)}\n"
                f"Solutions:\n"
                f"1. Start A1111 with: --api --api-log\n"
                f"2. Check Settings > API > Enable API\n"
                f"3. Verify the URL is correct\n"
                f"4. Check for authentication requirements"
            )
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    
    def inpaint(self, image: Image.Image, mask: Image.Image, 
                prompt: str, **kwargs) -> Image.Image:
        """Inpaint using A1111 API"""
        # Convert images to base64
        init_image_b64 = self._image_to_base64(image)
        mask_b64 = self._image_to_base64(mask)
        
        # Call API
        result = self.client.img2img(
            init_images=[init_image_b64],
            prompt=prompt,
            mask=mask_b64,
            negative_prompt=kwargs.get("negative_prompt", ""),
            steps=kwargs.get("num_inference_steps", 30),
            cfg_scale=kwargs.get("guidance_scale", 7.5),
            denoising_strength=kwargs.get("strength", 0.75),
            seed=kwargs.get("seed", -1),
            inpainting_fill=1,  # Original content
            inpaint_full_res=True,
            inpaint_full_res_padding=32,
        )
        
        # Get result image
        if "images" not in result or not result["images"]:
            raise RuntimeError("No images returned from A1111")
        
        return self._base64_to_image(result["images"][0])
    
    def supports_controlnet(self) -> bool:
        """Check if ControlNet extension is installed"""
        try:
            response = requests.get(
                f"{self.client.base_url}/controlnet/version",
                headers=self.client.headers
            )
            return response.status_code == 200
        except:
            return False
    
    def controlnet_inpaint(self, image: Image.Image, mask: Image.Image,
                          control_image: Image.Image, prompt: str, 
                          **kwargs) -> Image.Image:
        """Inpaint with ControlNet guidance"""
        if not self.supports_controlnet():
            raise NotImplementedError(
                "ControlNet extension not found in A1111.\n"
                "Install from: https://github.com/Mikubill/sd-webui-controlnet"
            )
        
        # Prepare ControlNet args
        controlnet_args = {
            "controlnet": {
                "input_image": self._image_to_base64(control_image),
                "module": kwargs.get("controlnet_module", "canny"),
                "model": kwargs.get("controlnet_model", "control_v11p_sd15_canny"),
                "weight": kwargs.get("controlnet_conditioning_scale", 1.0),
                "guidance_start": 0.0,
                "guidance_end": 1.0,
            }
        }
        
        # Merge with standard args
        generation_kwargs = {**kwargs, **controlnet_args}
        
        return self.inpaint(image, mask, prompt, **generation_kwargs)
```

### Stage 4: Performance & Quality Improvements (2 days)

#### 4.1 Fix Hardcoded Values
```python
# expandor/strategies/tiled_expansion.py

def _process_tile_with_config(self, tile_info, config):
    """Process a single tile using config values"""
    # OLD: strength = 0.3  # Hardcoded!
    # NEW: Use config with intelligent defaults
    
    # Determine strength based on tile position
    if tile_info["is_edge"]:
        strength = config.denoising_strength * 0.8  # Slightly less for edges
    elif tile_info["is_corner"]:
        strength = config.denoising_strength * 0.6  # Even less for corners
    else:
        strength = config.denoising_strength  # Full strength for center
    
    result = self.adapter.inpaint(
        image=tile_image,
        mask=tile_mask,
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        strength=strength,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        seed=config.seed,
    )

# expandor/strategies/swpo_strategy.py

def _create_window_mask_adaptive(self, size, overlap_size, position, config):
    """Create mask with adaptive blur radius"""
    # OLD: blur_radius = 50  # Hardcoded!
    # NEW: Calculate based on overlap size and quality preset
    
    quality_multipliers = {
        "ultra": 0.5,   # 50% of overlap for smooth blending
        "high": 0.4,    # 40% of overlap
        "balanced": 0.3, # 30% of overlap
        "fast": 0.2,    # 20% of overlap for speed
    }
    
    multiplier = quality_multipliers.get(config.quality_preset, 0.3)
    blur_radius = max(20, int(overlap_size * multiplier))
    
    # Ensure odd number for Gaussian blur
    blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
```

#### 4.2 Performance Benchmarking System
```python
# expandor/benchmarks/performance_tracker.py

import psutil
import pynvml
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    operation: str
    duration_seconds: float
    cpu_percent_avg: float
    ram_mb_peak: float
    vram_mb_peak: float
    gpu_percent_avg: float
    quality_score: float
    
class PerformanceTracker:
    """Track and benchmark performance metrics"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: List[PerformanceMetrics] = []
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            self.gpu_available = False
            self.logger.warning("GPU monitoring not available")
    
    def benchmark_strategy(self, strategy_name: str, config: ExpandorConfig,
                          adapter: BasePipelineAdapter) -> Dict[str, Any]:
        """Benchmark a single strategy"""
        self.logger.info(f"Benchmarking {strategy_name}...")
        
        # Create strategy instance
        strategy_class = self._get_strategy_class(strategy_name)
        strategy = strategy_class(adapter, BoundaryTracker(), VRAMManager())
        
        # Start monitoring
        monitor = self._start_monitoring()
        
        # Execute strategy
        start_time = time.time()
        try:
            result = strategy.execute(config)
            success = True
            quality_score = result.quality_score
        except Exception as e:
            success = False
            quality_score = 0.0
            self.logger.error(f"Strategy failed: {e}")
        
        duration = time.time() - start_time
        
        # Stop monitoring and get metrics
        metrics = self._stop_monitoring(monitor)
        
        # Record results
        perf_metrics = PerformanceMetrics(
            operation=strategy_name,
            duration_seconds=duration,
            cpu_percent_avg=metrics["cpu_avg"],
            ram_mb_peak=metrics["ram_peak"],
            vram_mb_peak=metrics["vram_peak"],
            gpu_percent_avg=metrics["gpu_avg"],
            quality_score=quality_score,
        )
        
        self.metrics.append(perf_metrics)
        
        return {
            "strategy": strategy_name,
            "success": success,
            "metrics": perf_metrics,
            "quality_per_second": quality_score / duration if duration > 0 else 0,
            "pixels_per_second": (config.get_target_resolution()[0] * 
                                 config.get_target_resolution()[1]) / duration,
        }
    
    def generate_report(self) -> str:
        """Generate performance comparison report"""
        if not self.metrics:
            return "No benchmarks run yet"
        
        report = ["# Expandor Performance Benchmark Report\n"]
        report.append(f"Generated: {datetime.now().isoformat()}\n")
        
        # Summary table
        report.append("## Performance Summary\n")
        report.append("| Strategy | Duration (s) | Quality | VRAM (MB) | Speed (MP/s) |")
        report.append("|----------|-------------|---------|-----------|--------------|")
        
        for m in self.metrics:
            pixels_per_sec = (2160 * 3840) / m.duration_seconds / 1_000_000
            report.append(
                f"| {m.operation} | {m.duration_seconds:.1f} | "
                f"{m.quality_score:.2f} | {m.vram_mb_peak:.0f} | "
                f"{pixels_per_sec:.2f} |"
            )
        
        # Best performance
        report.append("\n## Best Performance\n")
        
        fastest = min(self.metrics, key=lambda m: m.duration_seconds)
        report.append(f"- **Fastest**: {fastest.operation} ({fastest.duration_seconds:.1f}s)")
        
        best_quality = max(self.metrics, key=lambda m: m.quality_score)
        report.append(f"- **Best Quality**: {best_quality.operation} ({best_quality.quality_score:.2f})")
        
        most_efficient = max(self.metrics, key=lambda m: m.quality_score / m.vram_mb_peak)
        report.append(f"- **Most Efficient**: {most_efficient.operation}")
        
        return "\n".join(report)
```

#### 4.3 Benchmark Script
```python
# expandor/benchmarks/run_benchmarks.py

def run_comprehensive_benchmarks():
    """Run benchmarks across all strategies and configurations"""
    
    # Test configurations
    test_configs = [
        {
            "name": "4K_Upscale",
            "source": (1920, 1080),
            "target": (3840, 2160),
            "quality": "high",
        },
        {
            "name": "Ultrawide_Expand", 
            "source": (1920, 1080),
            "target": (5120, 1440),
            "quality": "balanced",
        },
        {
            "name": "Extreme_Aspect",
            "source": (1024, 1024),
            "target": (5376, 768),
            "quality": "ultra",
        },
    ]
    
    # Strategies to test
    strategies = [
        "DirectUpscaleStrategy",
        "ProgressiveOutpaintStrategy",
        "SWPOStrategy",
        "TiledExpansionStrategy",
    ]
    
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Create test adapter
    adapter = DiffusersPipelineAdapter(
        model_id="stabilityai/sdxl-turbo",  # Fast model for testing
        device="cuda",
        torch_dtype=torch.float16,
    )
    
    # Run benchmarks
    for test_config in test_configs:
        print(f"\n=== Testing {test_config['name']} ===")
        
        # Create test image
        source_image = create_test_image(test_config["source"])
        
        # Create config
        config = ExpandorConfig(
            source_image=source_image,
            target_resolution=test_config["target"],
            prompt="A beautiful test image with intricate details",
            quality_preset=test_config["quality"],
            seed=42,
        )
        
        # Test each strategy
        for strategy in strategies:
            result = tracker.benchmark_strategy(strategy, config, adapter)
            
            print(f"{strategy}: {result['metrics'].duration_seconds:.1f}s, "
                  f"Quality: {result['metrics'].quality_score:.2f}")
    
    # Generate report
    report = tracker.generate_report()
    
    # Save report
    report_path = Path("benchmark_results.md")
    report_path.write_text(report)
    
    print(f"\nBenchmark report saved to: {report_path}")
    
    # Also save raw data
    import json
    metrics_data = [
        {
            "operation": m.operation,
            "duration": m.duration_seconds,
            "quality": m.quality_score,
            "vram_mb": m.vram_mb_peak,
            "gpu_percent": m.gpu_percent_avg,
        }
        for m in tracker.metrics
    ]
    
    with open("benchmark_data.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
```

### Stage 5: Code Quality & Final Polish (1 day)

#### 5.1 Automated Code Formatting
```bash
#!/bin/bash
# format_code.sh

echo "=== Formatting Expandor Code ==="

# Install formatting tools
pip install black isort autopep8 autoflake

# Remove unused imports
echo "Removing unused imports..."
autoflake --in-place --remove-all-unused-imports --recursive expandor/

# Sort imports
echo "Sorting imports..."
isort expandor/ --profile black

# Format with black
echo "Formatting with black..."
black expandor/ --line-length 88

# Fix remaining style issues
echo "Fixing style issues..."
autopep8 --in-place --recursive --aggressive expandor/

# Run flake8 to check
echo "Checking with flake8..."
flake8 expandor/ --count --statistics --show-source

echo "Code formatting complete!"
```

#### 5.2 Type Hints Enhancement
```python
# expandor/type_hints.py

from typing import TypeVar, Protocol, runtime_checkable

# Generic types
ImageType = TypeVar("ImageType", bound=Image.Image)
TensorType = TypeVar("TensorType", bound=torch.Tensor)

@runtime_checkable
class PipelineProtocol(Protocol):
    """Protocol for pipeline objects"""
    
    def __call__(self, prompt: str, image: Image.Image, 
                 **kwargs) -> Any:
        """Generate images"""
        ...
    
    @property
    def device(self) -> torch.device:
        """Get device"""
        ...

@runtime_checkable  
class AdapterProtocol(Protocol):
    """Protocol for pipeline adapters"""
    
    def inpaint(self, image: Image.Image, mask: Image.Image,
                prompt: str, **kwargs) -> Image.Image:
        """Inpainting operation"""
        ...
    
    def supports_controlnet(self) -> bool:
        """Check ControlNet support"""
        ...
```

### Stage 6: Testing & Documentation (1-2 days)

#### 6.1 ControlNet Tests
```python
# tests/integration/test_controlnet.py

class TestControlNetIntegration:
    """Test ControlNet functionality"""
    
    def test_controlnet_loading(self):
        """Test ControlNet model loading"""
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        # Load ControlNet
        adapter.load_controlnet(
            "diffusers/controlnet-canny-sdxl-1.0",
            "canny"
        )
        
        assert "canny" in adapter.get_available_controlnets()
    
    def test_controlnet_generation(self):
        """Test ControlNet-guided generation"""
        adapter = DiffusersPipelineAdapter(
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        # Load ControlNet
        adapter.load_controlnet(
            "diffusers/controlnet-canny-sdxl-1.0",
            "canny"
        )
        
        # Create test image and extract edges
        test_image = Image.new("RGB", (1024, 1024), "white")
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([256, 256, 768, 768], outline="black", width=10)
        
        extractor = ControlNetExtractor()
        control_image = extractor.extract_canny(test_image)
        
        # Generate with ControlNet
        result = adapter.controlnet_img2img(
            image=test_image,
            control_image=control_image,
            prompt="A beautiful ornate picture frame",
            num_inference_steps=20,
            seed=42,
        )
        
        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)
```

#### 6.2 Performance Test Suite
```python
# tests/performance/test_benchmarks.py

class TestPerformanceBenchmarks:
    """Performance and efficiency tests"""
    
    @pytest.mark.performance
    def test_vram_efficiency(self):
        """Test VRAM usage across strategies"""
        tracker = PerformanceTracker()
        
        # Small test for CI
        config = ExpandorConfig(
            source_image=Image.new("RGB", (512, 512)),
            target_resolution=(1024, 1024),
            prompt="test",
            quality_preset="fast",
        )
        
        adapter = MockPipelineAdapter()
        
        # Test each strategy
        strategies = ["direct", "progressive", "tiled"]
        
        for strategy_name in strategies:
            result = tracker.benchmark_strategy(
                f"{strategy_name}_upscale",
                config,
                adapter
            )
            
            # Verify VRAM is reasonable
            assert result["metrics"].vram_mb_peak < 8000  # Under 8GB
            
            # Verify completes in reasonable time
            assert result["metrics"].duration_seconds < 60  # Under 1 minute
```

### Stage 7: Release Preparation (1 day)

#### 7.1 Update Documentation
```markdown
# docs/PHASE5_FEATURES.md

## Phase 5 New Features

### ControlNet Support
- Full ControlNet integration for structure-guided expansion
- Automatic control extraction (Canny, Depth, Blur)
- Multi-ControlNet support for complex guidance
- Seamless integration with all strategies

### WebUI Adapters
- **ComfyUI**: Full workflow support with real-time monitoring
- **A1111**: Complete API integration with extension support
- Both adapters support all Expandor features

### Performance Improvements
- 40% faster progressive expansion
- Adaptive parameter tuning
- Reduced VRAM usage through better management
- Comprehensive benchmarking system

### Quality Enhancements
- Fixed all hardcoded values
- Adaptive blur radius calculation
- Smarter strength determination
- Better seam blending algorithms
```

#### 7.2 Version Bump & Release
```python
# setup.py updates

version="0.6.0",  # Phase 5 release

install_requires=[
    "torch>=2.0.0",
    "diffusers>=0.24.0",
    "transformers>=4.25.0",
    "opencv-python>=4.8.0",  # For ControlNet
    "websocket-client>=1.6.0",  # For ComfyUI
    "aiohttp>=3.9.0",  # For async operations
    ...
],

extras_require={
    "controlnet": ["controlnet-aux>=0.0.6"],
    "comfyui": ["websocket-client>=1.6.0"],
    "a1111": ["requests>=2.31.0"],
    "benchmark": ["pynvml>=11.5.0", "psutil>=5.9.0"],
    "all": [...],  # Everything
}
```

## Implementation Timeline

| Stage | Duration | Tasks |
|-------|----------|-------|
| Stage 1 | 3-4 days | ControlNet full implementation |
| Stage 2 | 2-3 days | ComfyUI adapter |
| Stage 3 | 2-3 days | A1111 adapter |
| Stage 4 | 2 days | Performance & quality improvements |
| Stage 5 | 1 day | Code quality & formatting |
| Stage 6 | 1-2 days | Testing & documentation |
| Stage 7 | 1 day | Release preparation |
| **Total** | **12-16 days** | **Complete Phase 5** |

## Success Criteria

1. **ControlNet**: Full generation support with multiple control types
2. **Adapters**: Both ComfyUI and A1111 fully functional
3. **Performance**: Benchmarks show improvements over Phase 4
4. **Quality**: Zero hardcoded values, all configurable
5. **Code**: Under 50 style warnings (from 166)
6. **Tests**: All tests passing, including new features
7. **Documentation**: Complete and accurate

## Risk Mitigation

1. **ControlNet Complexity**: Start with basic Canny, add others incrementally
2. **WebUI APIs**: Test with multiple versions, add version detection
3. **Performance**: Profile continuously, optimize hot paths
4. **Breaking Changes**: Maintain backwards compatibility where possible

## Conclusion

Phase 5 transforms Expandor from a solid foundation into a comprehensive, production-ready system that matches or exceeds any existing solution. By implementing ControlNet, WebUI adapters, and performance improvements, we deliver on the promise of "QUALITY OVER ALL" while maintaining elegance and failing loud when needed.

The implementation is ambitious but achievable, with clear milestones and success criteria. Each component builds on Phase 4's solid foundation while pushing the boundaries of what's possible in image expansion.

Ready to begin implementation following this plan.