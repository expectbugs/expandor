"""
Pipeline orchestration with fallback support and comprehensive error handling
"""

import time
import traceback
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

from .exceptions import ExpandorError, VRAMError, StrategyError
from .result import ExpandorResult, StageResult
from .metadata_tracker import MetadataTracker
from .boundary_tracker import BoundaryTracker
from ..strategies.base_strategy import BaseExpansionStrategy
from ..strategies import get_strategy_class

class PipelineOrchestrator:
    """
    Executes expansion strategies with fallback support
    
    This class is responsible for:
    1. Executing strategies with proper error handling
    2. Managing fallback chains when strategies fail
    3. Coordinating between strategy, metadata, and boundary tracking
    4. Ensuring atomic operations (all-or-nothing)
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize orchestrator
        
        Args:
            config: Global configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline_registry = {}
        
        # Load fallback configuration
        self.fallback_config = config.get('vram_strategies', {}).get('fallback_chain', {
            1: 'tiled_large',
            2: 'tiled_medium',
            3: 'tiled_small',
            4: 'cpu_offload'
        })
        
        # Track execution state
        self.current_strategy = None
        self.execution_history = []
    
    def execute(self,
                strategy: BaseExpansionStrategy,
                config,
                metadata_tracker: MetadataTracker,
                boundary_tracker: BoundaryTracker,
                temp_dir: Optional[Path] = None) -> ExpandorResult:
        """
        Execute strategy with comprehensive error handling and fallbacks
        
        Args:
            strategy: Primary strategy to execute
            config: ExpandorConfig instance
            metadata_tracker: Metadata tracking instance
            boundary_tracker: Boundary tracking instance
            
        Returns:
            ExpandorResult with processed image and metadata
            
        Raises:
            ExpandorError: If all strategies fail
        """
        # Track start
        execution_start = time.time()
        self.execution_history.clear()
        
        # Prepare strategy with pipelines and trackers
        self._prepare_strategy(strategy, config, boundary_tracker, metadata_tracker)
        
        # Build fallback chain
        fallback_chain = self._build_fallback_chain(strategy, config)
        
        # Execute with fallbacks
        last_error = None
        fallback_count = 0
        
        for attempt, current_strategy in enumerate(fallback_chain):
            try:
                self.current_strategy = current_strategy
                strategy_name = current_strategy.__class__.__name__
                
                self.logger.info(
                    f"Executing {strategy_name} "
                    f"{'(primary)' if attempt == 0 else f'(fallback {attempt})'}"
                )
                
                # Record execution attempt
                metadata_tracker.record_event("strategy_execution_start", {
                    "strategy": strategy_name,
                    "attempt": attempt + 1,
                    "is_fallback": attempt > 0
                })
                
                # Execute strategy
                result = self._execute_single_strategy(
                    current_strategy,
                    config,
                    metadata_tracker,
                    boundary_tracker
                )
                
                # Success! Update result metadata
                result.strategy_used = strategy_name
                result.fallback_count = fallback_count
                result.metadata["execution_history"] = self.execution_history
                
                # Record success
                metadata_tracker.record_event("strategy_execution_success", {
                    "strategy": strategy_name,
                    "duration": time.time() - execution_start
                })
                
                return result
                
            except Exception as e:
                # Record failure
                error_info = {
                    "strategy": strategy_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
                self.execution_history.append({
                    "strategy": strategy_name,
                    "status": "failed",
                    "error": str(e),
                    "attempt": attempt + 1
                })
                
                metadata_tracker.record_event("strategy_execution_failed", error_info)
                
                # Log based on error type
                if isinstance(e, VRAMError):
                    self.logger.warning(f"{strategy_name} failed due to VRAM: {e}")
                else:
                    self.logger.error(f"{strategy_name} failed: {e}")
                
                last_error = e
                fallback_count += 1
                
                # Check if we should continue with fallbacks
                if attempt == len(fallback_chain) - 1:
                    # This was the last strategy
                    break
                
                if not self._should_continue_fallback(e, config):
                    break
                
                # Prepare for next attempt
                self.logger.info("Attempting fallback strategy...")
                time.sleep(0.5)  # Brief pause before retry
        
        # All strategies failed - fail loud
        self._handle_complete_failure(last_error, self.execution_history)
    
    def register_pipeline(self, name: str, pipeline: Any):
        """Register a pipeline for strategies to use"""
        self.pipeline_registry[name] = pipeline
        self.logger.debug(f"Registered pipeline: {name}")
    
    def _prepare_strategy(self, 
                         strategy: BaseExpansionStrategy, 
                         config,
                         boundary_tracker: BoundaryTracker,
                         metadata_tracker: MetadataTracker):
        """
        Prepare strategy with necessary components
        
        Args:
            strategy: Strategy to prepare
            config: ExpandorConfig
            boundary_tracker: Boundary tracker instance
            metadata_tracker: Metadata tracker instance
        """
        # Inject pipelines from config and registry
        if hasattr(config, 'inpaint_pipeline') and config.inpaint_pipeline:
            strategy.inpaint_pipeline = config.inpaint_pipeline
        elif 'inpaint' in self.pipeline_registry:
            strategy.inpaint_pipeline = self.pipeline_registry['inpaint']
            
        if hasattr(config, 'refiner_pipeline') and config.refiner_pipeline:
            strategy.refiner_pipeline = config.refiner_pipeline
        elif 'refiner' in self.pipeline_registry:
            strategy.refiner_pipeline = self.pipeline_registry['refiner']
            
        if hasattr(config, 'img2img_pipeline') and config.img2img_pipeline:
            strategy.img2img_pipeline = config.img2img_pipeline
        elif 'img2img' in self.pipeline_registry:
            strategy.img2img_pipeline = self.pipeline_registry['img2img']
        
        # Inject trackers
        strategy.boundary_tracker = boundary_tracker
        strategy.metadata_tracker = metadata_tracker
        
        # Set logger if not already set
        if not hasattr(strategy, 'logger') or strategy.logger is None:
            strategy.logger = self.logger
        
        # Validate strategy has required pipelines
        strategy.validate_requirements()
    
    def _execute_single_strategy(self,
                                strategy: BaseExpansionStrategy,
                                config,
                                metadata_tracker: MetadataTracker,
                                boundary_tracker: BoundaryTracker) -> ExpandorResult:
        """
        Execute a single strategy with proper tracking
        
        This method ensures all tracking is properly updated during execution
        """
        # Create execution context
        # This context is passed to strategies to provide access to shared resources
        context = {
            "config": config,                          # Original ExpandorConfig
            "metadata_tracker": metadata_tracker,      # For recording events/metrics
            "boundary_tracker": boundary_tracker,      # For tracking expansion boundaries
            "pipeline_registry": self.pipeline_registry,  # Available AI pipelines
            "stage_callback": lambda stage: self._record_stage(stage, metadata_tracker),  # Stage recording function
            "save_stages": getattr(config, 'save_stages', False),         # Whether to save intermediate results
            "stage_dir": getattr(config, 'stage_dir', Path('temp/stages')),  # Directory for saving stages
            "temp_dir": temp_dir or Path("temp")      # Temp directory for intermediate files
        }
        
        # Execute strategy
        try:
            # Update metadata tracker stage
            metadata_tracker.enter_stage(f"executing_{strategy.__class__.__name__}")
            
            # Execute with context
            raw_result = strategy.execute(config, context)
            
            # Mark stage complete
            metadata_tracker.exit_stage(success=True)
            
            # Convert to ExpandorResult if needed
            if isinstance(raw_result, ExpandorResult):
                return raw_result
            else:
                return self._convert_to_result(raw_result, strategy, config)
                
        except Exception as e:
            # Mark stage failed
            metadata_tracker.exit_stage(success=False, error=str(e))
            
            # Ensure strategy cleanup on error
            if hasattr(strategy, 'cleanup'):
                try:
                    strategy.cleanup()
                except Exception as cleanup_error:
                    self.logger.debug(f"Cleanup error after execution failure: {cleanup_error}")
                    # Don't let cleanup errors mask the real error
            raise
    
    def _build_fallback_chain(self, 
                             primary_strategy: BaseExpansionStrategy,
                             config) -> List[BaseExpansionStrategy]:
        """
        Build chain of strategies to try in order
        
        Returns list with primary strategy first, then fallbacks
        """
        chain = [primary_strategy]
        
        # Check if fallbacks are disabled
        if config.quality_preset == 'fast':
            return chain  # No fallbacks in fast mode
        
        # Get strategy name
        primary_name = primary_strategy.__class__.__name__.replace('Strategy', '').lower()
        
        # Add configured fallbacks
        # Map fallback config to strategy classes
        fallback_map = {
            'tiled_large': ('tiled_expansion', {'tile_size': 1024}),
            'tiled_medium': ('tiled_expansion', {'tile_size': 768}),
            'tiled_small': ('tiled_expansion', {'tile_size': 512}),
            'cpu_offload': ('cpu_offload', {})
        }
        
        # Build fallback chain based on config
        for priority in sorted(self.fallback_config.keys()):
            fallback_name = self.fallback_config[priority]
            
            if fallback_name in fallback_map:
                strategy_name, extra_config = fallback_map[fallback_name]
                
                # Skip if same as primary
                if strategy_name == primary_name:
                    continue
                
                # Skip if disabled
                if strategy_name == 'tiled_expansion' and not getattr(config, 'allow_tiled', True):
                    continue
                if strategy_name == 'cpu_offload' and not getattr(config, 'allow_cpu_offload', True):
                    continue
                
                # Create fallback strategy
                try:
                    # Load strategy class
                    strategy_class = get_strategy_class(strategy_name)
                    
                    # Create instance with extra config
                    fallback = strategy_class(
                        config={**self.config.get('strategies', {}).get(strategy_name, {}), 
                               **extra_config},
                        logger=self.logger
                    )
                    
                    # Prepare it
                    self._prepare_strategy(fallback, config, 
                                         primary_strategy.boundary_tracker,
                                         primary_strategy.metadata_tracker)
                    
                    chain.append(fallback)
                    
                except Exception as e:
                    self.logger.warning(f"Could not create fallback {strategy_name}: {e}")
        
        return chain
    
    def _should_continue_fallback(self, error: Exception, config) -> bool:
        """
        Determine if we should try fallback after an error
        
        Some errors are unrecoverable and fallbacks won't help
        """
        # Don't retry on configuration errors
        if isinstance(error, (ValueError, TypeError)):
            return False
        
        # Don't retry on file errors
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return False
        
        # Always retry VRAM errors if fallbacks available
        if isinstance(error, VRAMError):
            return True
        
        # Retry other errors based on quality preset
        if config.quality_preset in ['ultra', 'high']:
            return True  # Try hard to succeed
        
        return False
    
    def _record_stage(self, stage: StageResult, metadata_tracker: MetadataTracker):
        """Record a stage completion"""
        metadata_tracker.record_event("stage_complete", {
            "name": stage.name,
            "method": stage.method,
            "input_size": stage.input_size,
            "output_size": stage.output_size,
            "duration": stage.duration_seconds,
            "vram_used": stage.vram_used_mb,
            "artifacts_detected": stage.artifacts_detected
        })
    
    def _convert_to_result(self, 
                          raw_result: Dict[str, Any],
                          strategy: BaseExpansionStrategy,
                          config) -> ExpandorResult:
        """Convert strategy output to ExpandorResult"""
        # Extract required fields
        image_path = raw_result.get('image_path')
        if not image_path:
            raise StrategyError(f"{strategy.__class__.__name__} did not return image_path")
        
        # Create result
        result = ExpandorResult(
            image_path=Path(image_path),
            size=raw_result.get('size', (0, 0)),
            success=True,
            stages=raw_result.get('stages', []),
            boundaries=raw_result.get('boundaries', []),
            strategy_used=strategy.__class__.__name__,
            metadata=raw_result.get('metadata', {})
        )
        
        # Copy stage results if from strategy
        if hasattr(strategy, 'stage_results'):
            result.stages = strategy.stage_results
        
        return result
    
    def _handle_complete_failure(self, last_error: Exception, history: List[Dict]):
        """Handle case where all strategies failed"""
        # Build comprehensive error message
        error_lines = [
            "All expansion strategies failed!",
            f"Attempted {len(history)} strategies:",
            ""
        ]
        
        for item in history:
            error_lines.append(
                f"  - {item['strategy']} (attempt {item['attempt']}): {item['error']}"
            )
        
        error_lines.extend([
            "",
            "This indicates a critical issue that prevents image expansion.",
            "Possible causes:",
            "  1. Insufficient system resources (RAM/VRAM)",
            "  2. Corrupted input image",  
            "  3. Invalid configuration",
            "  4. Missing required pipelines",
            "",
            "Last error details:"
        ])
        
        error_message = "\n".join(error_lines)
        
        # Raise comprehensive error
        raise ExpandorError(
            error_message,
            stage="pipeline_execution",
            partial_result={"execution_history": history}
        ) from last_error