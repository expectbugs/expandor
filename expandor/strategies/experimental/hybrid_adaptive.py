"""
Hybrid Adaptive Strategy
Intelligently combines multiple strategies based on analysis.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ...core.config import ExpandorConfig
from ...core.exceptions import StrategyError
from ...processors.edge_analysis import EdgeAnalyzer
from ...processors.refinement.smart_refiner import SmartRefiner
from ...utils.dimension_calculator import DimensionCalculator
from ..base_strategy import BaseExpansionStrategy
from ..cpu_offload import CPUOffloadStrategy
from ..direct_upscale import DirectUpscaleStrategy
# Import other strategies for delegation
from ..progressive_outpaint import ProgressiveOutpaintStrategy
from ..swpo_strategy import SWPOStrategy


@dataclass
class HybridPlan:
    """Execution plan for hybrid strategy"""

    steps: List[Dict[str, Any]]
    estimated_vram: float
    estimated_quality: float
    rationale: str


class HybridAdaptiveStrategy(BaseExpansionStrategy):
    """
    Intelligently combines multiple strategies based on:
    - Input/output characteristics
    - Available resources
    - Quality requirements
    - Optimal path analysis

    Can delegate to:
    - Direct upscale for simple scaling
    - Progressive outpaint for moderate aspect changes
    - SWPO for extreme aspect ratios
    - CPU offload when memory constrained
    - Combinations for complex scenarios
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize hybrid adaptive strategy."""
        super().__init__(config=config, metrics=metrics, logger=logger)

        # Initialize components
        self.dimension_calc = DimensionCalculator(self.logger)

        # Initialize sub-strategies (they'll be properly configured later)
        self.strategies = {
            "progressive": ProgressiveOutpaintStrategy(
                config, metrics, logger), "direct": DirectUpscaleStrategy(
                config, metrics, logger), "swpo": SWPOStrategy(
                config, metrics, logger), "cpu_offload": CPUOffloadStrategy(
                    config, metrics, logger), }

        # Smart refiner for quality improvement
        self.smart_refiner = SmartRefiner(logger=self.logger)

        # Load strategy-specific config using ConfigurationManager
        from ...core.configuration_manager import ConfigurationManager
        self.config_manager = ConfigurationManager()
        
        # Get hybrid_adaptive strategy config
        self.strategy_params = self.config_manager.get_value('strategies.hybrid_adaptive', {})

        # FAIL LOUD if required params missing
        required = [
            'aspect_ratio_threshold',
            'extreme_ratio_threshold',
            'vram_safety_factor']
        missing = [p for p in required if p not in self.strategy_params]
        if missing:
            raise ValueError(
                f"HybridAdaptiveStrategy missing required parameters: {missing}\n"
                f"Please check config/strategy_parameters.yaml"
            )

        # Decision thresholds from config
        # 20% change triggers outpainting
        self.aspect_ratio_threshold = self.strategy_params['aspect_ratio_threshold']
        # 3x+ ratio triggers SWPO
        self.extreme_ratio_threshold = self.strategy_params['extreme_ratio_threshold']
        # Use 80% of available VRAM
        self.vram_safety_factor = self.strategy_params['vram_safety_factor']

    def validate_requirements(self):
        """Validate that at least one sub-strategy can work."""
        # At least one pipeline must be available
        pass  # Checked in execute

    def estimate_vram(self, config: ExpandorConfig) -> Dict[str, float]:
        """
        Estimate VRAM based on planned approach.
        """
        # Analyze the expansion
        plan = self._analyze_expansion(config)

        # Get estimate from primary strategy
        primary_strategy = self.strategies.get(plan.steps[0]["strategy"])
        if primary_strategy:
            return primary_strategy.estimate_vram(config)

        # Fallback estimate
        return super().estimate_vram(config)

    def execute(
        self, config: ExpandorConfig, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute hybrid adaptive strategy.

        Analyzes the task and delegates to optimal strategy or combination.
        """
        self._context = context or {}
        start_time = time.time()

        self.logger.info("Starting Hybrid Adaptive expansion strategy")

        # Validate inputs
        self.validate_inputs(config)

        # Check available pipelines
        available_pipelines = []
        if self.inpaint_pipeline:
            available_pipelines.append("inpaint")
        if self.img2img_pipeline:
            available_pipelines.append("img2img")

        if not available_pipelines:
            raise StrategyError(
                "Hybrid adaptive requires at least one pipeline")

        try:
            # Analyze expansion and create plan
            plan = self._analyze_expansion(config)

            self.logger.info(f"Hybrid plan: {plan.rationale}")
            self.logger.info(f"Steps: {[s['name'] for s in plan.steps]}")

            # Check VRAM availability
            if plan.estimated_vram > 0:
                available_vram = self.vram_manager.get_available_vram() or 0
                if plan.estimated_vram > available_vram * self.vram_safety_factor:
                    # Switch to CPU offload
                    self.logger.warning(
                        f"Insufficient VRAM ({available_vram}MB), switching to CPU offload")
                    plan = self._create_cpu_offload_plan(config)

            # Execute plan
            current_result = None

            for i, step in enumerate(plan.steps):
                step_start = time.time()

                self.logger.info(
                    f"Executing step {i + 1}/{len(plan.steps)}: {step['name']}"
                )

                # Prepare step config
                step_config = self._prepare_step_config(
                    config, step, current_result)

                # Get strategy
                strategy = self.strategies.get(step["strategy"])
                if not strategy:
                    raise StrategyError(
                        f"Unknown strategy: {
                            step['strategy']}"
                    )

                # Inject dependencies
                strategy.boundary_tracker = self.boundary_tracker
                strategy.metadata_tracker = self.metadata_tracker
                strategy.vram_manager = self.vram_manager

                # Inject pipelines from parent
                if hasattr(self, 'inpaint_pipeline') and self.inpaint_pipeline:
                    strategy.inpaint_pipeline = self.inpaint_pipeline
                if hasattr(self, 'img2img_pipeline') and self.img2img_pipeline:
                    strategy.img2img_pipeline = self.img2img_pipeline
                if hasattr(self, 'refiner_pipeline') and self.refiner_pipeline:
                    strategy.refiner_pipeline = self.refiner_pipeline

                # Execute step
                step_result = strategy.execute(step_config, context)

                # Record stage
                self.record_stage(
                    name=f"hybrid_{step['name']}",
                    method=step["strategy"],
                    input_size=(
                        step_config.source_image.size
                        if hasattr(step_config.source_image, "size")
                        else (0, 0)
                    ),
                    output_size=step_result["size"],
                    start_time=step_start,
                    metadata={
                        "step_index": i,
                        "substrategy": step["strategy"],
                        "step_config": step,
                    },
                )

                current_result = step_result

                # Save intermediate if requested
                if config.save_stages and config.stage_dir:
                    stage_path = config.stage_dir / f"hybrid_step_{i:02d}.png"
                    img = Image.open(step_result["image_path"])
                    # Get compression from config
                    from ...core.configuration_manager import ConfigurationManager
                    config_manager = ConfigurationManager()
                    png_compression = config_manager.get_value("output.formats.png.compression")
                    img.save(stage_path, "PNG", compress_level=png_compression)

            # Optional quality refinement
            # Auto-refine is now always enabled for quality
            if current_result:
                self.logger.info("Applying smart quality refinement")
                current_result = self._apply_quality_refinement(
                    current_result, config)

            # Final result
            return {
                "image_path": current_result["image_path"],
                "size": current_result["size"],
                "stages": self.stage_results,
                "boundaries": (
                    self.boundary_tracker.get_all_boundaries()
                    if self.boundary_tracker
                    else []
                ),
                "metadata": {
                    "strategy": "hybrid_adaptive",
                    "plan": plan.rationale,
                    "steps_executed": len(plan.steps),
                    "substrategy_sequence": [s["strategy"] for s in plan.steps],
                    "duration": time.time() - start_time,
                },
            }

        except Exception as e:
            # FAIL LOUD
            self.logger.error(f"Hybrid adaptive strategy failed: {str(e)}")
            raise StrategyError(
                f"Hybrid adaptive execution failed: {
                    str(e)}"
            ) from e
        finally:
            self.cleanup()

    def _analyze_expansion(self, config: ExpandorConfig) -> HybridPlan:
        """
        Analyze the expansion task and create optimal plan.
        """
        source_w, source_h = (
            config.source_image.size if hasattr(
                config.source_image, "size") else (
                0, 0))
        target_w, target_h = config.get_target_resolution()

        # Validate dimensions to prevent division by zero
        if source_w <= 0 or source_h <= 0:
            raise StrategyError(
                f"Invalid source dimensions: {source_w}x{source_h}. "
                f"Source image must have positive width and height."
            )
        
        if target_w <= 0 or target_h <= 0:
            raise StrategyError(
                f"Invalid target dimensions: {target_w}x{target_h}. "
                f"Target dimensions must be positive."
            )

        # Calculate metrics safely
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h
        aspect_change = abs(target_aspect - source_aspect) / source_aspect

        scale_x = target_w / source_w
        scale_y = target_h / source_h
        max_scale = max(scale_x, scale_y)

        extreme_ratio = max(target_w / target_h, target_h / target_w)

        # Determine approach
        steps = []

        # Get thresholds from config - FAIL LOUD
        simple_aspect_threshold = self.config_manager.get_value('strategies.hybrid_adaptive.simple_aspect_change_threshold')
        simple_scale_threshold = self.config_manager.get_value('strategies.hybrid_adaptive.simple_scale_difference_threshold')
        
        if (aspect_change < simple_aspect_threshold and 
            abs(scale_x - scale_y) < simple_scale_threshold):
            # Simple upscale - check if direct upscale is available
            try:
                self.strategies["direct"].validate_requirements()
                steps.append(
                    {
                        "name": "direct_upscale",
                        "strategy": "direct",
                        "config_overrides": {},
                    }
                )
                rationale = f"Simple {
                    max_scale:.1f}x upscale with minimal aspect change"
                estimated_vram = self.strategies["direct"].estimate_vram(config)[
                    "peak_vram_mb"
                ]
                estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.simple')
            except BaseException:
                # Fall back to progressive with img2img
                if self.img2img_pipeline:
                    steps.append(
                        {
                            "name": "progressive_img2img",
                            "strategy": "progressive",
                            "config_overrides": {},
                        }
                    )
                    rationale = f"Progressive {
                        max_scale:.1f}x upscale using img2img"
                    estimated_vram = self.strategies["progressive"].estimate_vram(
                        config
                    )["peak_vram_mb"]
                    estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.progressive')
                else:
                    raise StrategyError(
                        "No suitable upscaling method available")

        elif extreme_ratio > self.extreme_ratio_threshold or max(scale_x, scale_y) > self.config_manager.get_value('strategies.hybrid_adaptive.extreme_scale_threshold'):
            # Extreme expansion - use SWPO
            if self.inpaint_pipeline:
                steps.append(
                    {
                        "name": "swpo_expansion",
                        "strategy": "swpo",
                        "config_overrides": {
                            "window_size": self.config_manager.get_value('strategies.hybrid_adaptive.swpo_thresholds.window_size_large') if extreme_ratio > self.config_manager.get_value('strategies.hybrid_adaptive.swpo_thresholds.extreme_ratio') else self.config_manager.get_value('strategies.hybrid_adaptive.swpo_thresholds.window_size_normal'),
                            "overlap_ratio": self.config_manager.get_value('strategies.hybrid_adaptive.swpo_thresholds.overlap_high') if extreme_ratio > self.config_manager.get_value('strategies.hybrid_adaptive.swpo_thresholds.extreme_ratio') else self.config_manager.get_value('strategies.hybrid_adaptive.swpo_thresholds.overlap_normal'),
                        },
                    })
                rationale = f"Extreme {extreme_ratio:.1f}:1 ratio using SWPO"
                estimated_vram = self.strategies["swpo"].estimate_vram(config)[
                    "peak_vram_mb"
                ]
                estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.swpo')
            else:
                # Fallback to CPU offload
                steps.append(
                    {
                        "name": "cpu_tiled_expansion",
                        "strategy": "cpu_offload",
                        "config_overrides": {},
                    }
                )
                rationale = f"Extreme expansion using CPU offload (no inpaint pipeline)"
                estimated_vram = self.config_manager.get_value('strategies.hybrid_adaptive.vram_estimates.minimal')
                estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.cpu_fallback')

        elif aspect_change > self.aspect_ratio_threshold:
            # Moderate aspect change - progressive outpaint
            if self.inpaint_pipeline:
                steps.append(
                    {
                        "name": "progressive_expansion",
                        "strategy": "progressive",
                        "config_overrides": {},
                    }
                )

                # Check if we need additional upscaling after aspect adjustment
                # Progressive only adjusts aspect ratio, not final size
                additional_upscale_threshold = self.config_manager.get_value('strategies.hybrid_adaptive.additional_upscale_threshold')
                if max_scale > additional_upscale_threshold:  # Need additional scaling
                    steps.append(
                        {
                            "name": "final_upscale",
                            "strategy": "direct",
                            "config_overrides": {},
                        }
                    )
                    rationale = f"Progressive outpainting for {
                        aspect_change:.1%} aspect change + upscaling"
                    estimated_vram = max(
                        self.strategies["progressive"].estimate_vram(config)[
                            "peak_vram_mb"
                        ],
                        self.strategies["direct"].estimate_vram(config)["peak_vram_mb"],
                    )
                else:
                    rationale = f"Progressive outpainting for {
                        aspect_change:.1%} aspect change"
                    estimated_vram = self.strategies["progressive"].estimate_vram(
                        config
                    )["peak_vram_mb"]
                estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.progressive')
            else:
                # Use direct with post-processing
                steps.append(
                    {
                        "name": "direct_with_refine",
                        "strategy": "direct",
                        "config_overrides": {},
                    }
                )
                rationale = f"Direct upscale with refinement (no inpaint pipeline)"
                estimated_vram = self.strategies["direct"].estimate_vram(config)[
                    "peak_vram_mb"
                ]
                estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.tiled_low')

        else:
            # Default to progressive for safety
            steps.append(
                {
                    "name": "safe_progressive",
                    "strategy": "progressive",
                    "config_overrides": {"max_expansion_ratio": 1.3},
                }
            )
            rationale = "Safe progressive expansion"
            estimated_vram = self.strategies["progressive"].estimate_vram(config)[
                "peak_vram_mb"
            ]
            estimated_quality = self.config_manager.get_value('strategies.hybrid_adaptive.quality_estimates.final')

        return HybridPlan(
            steps=steps,
            estimated_vram=estimated_vram,
            estimated_quality=estimated_quality,
            rationale=rationale,
        )

    def _create_cpu_offload_plan(self, config: ExpandorConfig) -> HybridPlan:
        """Create plan using CPU offload for low memory."""
        return HybridPlan(
            steps=[
                {
                    "name": "memory_efficient_expansion",
                    "strategy": "cpu_offload",
                    "config_overrides": {"tile_size": 384, "overlap": 64},
                }
            ],
            estimated_vram=512,
            estimated_quality=0.7,
            rationale="Memory-efficient expansion using CPU offload",
        )

    def _prepare_step_config(
        self,
        base_config: ExpandorConfig,
        step: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]],
    ) -> ExpandorConfig:
        """Prepare configuration for a step."""
        # Create copy of base config
        step_config = ExpandorConfig(
            source_image=base_config.source_image,
            target_resolution=base_config.get_target_resolution(),
            prompt=base_config.prompt,
            negative_prompt=base_config.negative_prompt,
            seed=base_config.seed,
            strategy=base_config.strategy,
            quality_preset=base_config.quality_preset,
            denoising_strength=base_config.denoising_strength,
            guidance_scale=base_config.guidance_scale,
            num_inference_steps=base_config.num_inference_steps,
            save_stages=base_config.save_stages,
            stage_dir=base_config.stage_dir,
            verbose=base_config.verbose,
            vram_limit_mb=base_config.vram_limit_mb,
            use_cpu_offload=base_config.use_cpu_offload,
            window_size=base_config.window_size,
            overlap_ratio=base_config.overlap_ratio,
            tile_size=base_config.tile_size,
            source_metadata=base_config.source_metadata,
        )

        # Update source image from previous result
        if previous_result and "image_path" in previous_result:
            step_config.source_image = Image.open(
                previous_result["image_path"])

        # Apply step overrides
        # Validate config_overrides exists
        if "config_overrides" not in step:
            step["config_overrides"] = {}
        for key, value in step["config_overrides"].items():
            if hasattr(step_config, key):
                setattr(step_config, key, value)

        return step_config

    def _apply_quality_refinement(
        self, result: Dict[str, Any], config: ExpandorConfig
    ) -> Dict[str, Any]:
        """Apply smart quality refinement to result."""
        try:
            # Load image
            image = Image.open(result["image_path"])

            # Get boundaries for artifact detection
            # Validate boundaries exist in result
            if "boundaries" not in result:
                self.logger.warning("No boundaries returned from expansion process, using empty list")
                boundaries = []
            else:
                boundaries = result["boundaries"]

            # Detect artifacts
            analyzer = EdgeAnalyzer()
            analysis = analyzer.analyze_image(
                np.array(image),
                boundaries=(
                    [
                        {
                            "position": (
                                b["position"] if isinstance(b, dict) else b.position
                            ),
                            "direction": (
                                b["direction"] if isinstance(b, dict) else b.direction
                            ),
                        }
                        for b in boundaries
                    ]
                    if boundaries
                    else None
                ),
            )

            if analysis["artifacts"]:
                # Apply refinement
                refinement_result = self.smart_refiner.refine_image(
                    image=image,
                    artifacts=analysis["artifacts"],
                    pipeline=config.img2img_pipeline or config.inpaint_pipeline,
                    prompt=config.prompt,
                    boundaries=boundaries,
                )

                if refinement_result.success:
                    # Update result
                    result["image_path"] = refinement_result.image_path
                    result["metadata"]["quality_refined"] = True
                    result["metadata"][
                        "artifacts_fixed"
                    ] = refinement_result.regions_refined

        except Exception as e:
            self.logger.warning(f"Quality refinement failed: {e}")

        return result
