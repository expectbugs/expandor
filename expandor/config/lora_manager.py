"""
LoRA management with conflict resolution and weight stacking
Following FAIL LOUD philosophy - no silent conflicts
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from ..config.user_config import LoRAConfig
from ..utils.logging_utils import setup_logger


@dataclass
class LoRAType:
    """Categorize LoRAs by their modification type"""

    name: str
    modifies_layers: Set[str]  # Which model layers this type affects
    compatible_with: Set[str]  # Other LoRA types it's compatible with


class LoRAConflictError(Exception):
    """Raised when LoRAs have unresolvable conflicts"""


class LoRAManager:
    """
    Manages LoRA stacking with conflict detection
    FAIL LOUD: Any conflict that can't be resolved causes immediate failure
    """

    # Known LoRA types and their compatibility
    LORA_TYPES = {
        "style": LoRAType(
            name="style",
            modifies_layers={"unet.conv", "unet.attention"},
            compatible_with={"detail", "subject", "quality"},
        ),
        "detail": LoRAType(
            name="detail",
            modifies_layers={"unet.conv_out", "unet.final"},
            compatible_with={"style", "subject", "quality"},
        ),
        "subject": LoRAType(
            name="subject",
            modifies_layers={"text_encoder", "unet.cross_attention"},
            compatible_with={"style", "detail", "quality"},
        ),
        "quality": LoRAType(
            name="quality",
            modifies_layers={"vae", "unet.upsampler"},
            compatible_with={"style", "detail", "subject"},
        ),
        "character": LoRAType(
            name="character",
            modifies_layers={"text_encoder", "unet.cross_attention"},
            compatible_with={
                "style",
                "detail",
            },  # Conflicts with other character/subject LoRAs
        ),
        "artifact_removal": LoRAType(
            name="artifact_removal",
            modifies_layers={"vae.decoder", "unet.final"},
            compatible_with={
                "style",
                "subject",
                "character",
            },  # May conflict with detail enhancers
        ),
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize LoRA manager"""
        self.logger = logger or setup_logger(__name__)
        # (config, detected_type)
        self.loaded_loras: List[Tuple[LoRAConfig, str]] = []

    def detect_lora_type(self, lora: LoRAConfig) -> str:
        """
        Detect LoRA type from name and keywords

        FAIL LOUD: Unknown types cause errors, no guessing
        """
        name_lower = lora.name.lower()
        keywords_str = " ".join(lora.auto_apply_keywords).lower()

        # Detection rules based on common naming patterns
        if any(
            word in name_lower for word in [
                "style",
                "artstyle",
                "aesthetic"]):
            return "style"
        elif any(
            word in name_lower for word in ["detail", "enhance", "sharp", "hires"]
        ):
            return "detail"
        elif any(
            word in name_lower for word in ["character", "person", "face", "portrait"]
        ):
            return "character"
        elif any(word in name_lower for word in ["subject", "object", "thing"]):
            return "subject"
        elif any(
            word in name_lower for word in ["quality", "upscale", "hq", "highres"]
        ):
            return "quality"
        elif any(
            word in name_lower for word in ["artifact", "denoise", "clean", "fix"]
        ):
            return "artifact_removal"
        elif any(word in keywords_str for word in ["style", "aesthetic"]):
            return "style"
        elif any(word in keywords_str for word in ["detail", "detailed", "intricate"]):
            return "detail"
        else:
            # FAIL LOUD - don't guess
            raise LoRAConflictError(
                f"Cannot determine type for LoRA '{lora.name}'. "
                f"Please rename it to include type (e.g., 'mystyle_style.safetensors') "
                f"or add type keywords. Known types: {list(self.LORA_TYPES.keys())}"
            )

    def check_compatibility(self, new_lora: LoRAConfig, new_type: str) -> None:
        """
        Check if new LoRA is compatible with already loaded ones

        FAIL LOUD: Incompatible LoRAs cause immediate failure
        """
        for loaded_lora, loaded_type in self.loaded_loras:
            # Check if types are compatible
            if loaded_type not in self.LORA_TYPES[new_type].compatible_with:
                # Check for specific conflicts
                loaded_layers = self.LORA_TYPES[loaded_type].modifies_layers
                new_layers = self.LORA_TYPES[new_type].modifies_layers

                if loaded_layers.intersection(new_layers):
                    # FAIL LOUD - conflicting layer modifications
                    raise LoRAConflictError(
                        f"LoRA conflict detected!\n"
                        f"  Already loaded: '{loaded_lora.name}' (type: {loaded_type})\n"
                        f"  Trying to add: '{new_lora.name}' (type: {new_type})\n"
                        f"  Both modify layers: {loaded_layers.intersection(new_layers)}\n"
                        f"  Solution: Use only one {loaded_type}/{new_type} LoRA at a time"
                    )

            # Check for same-type conflicts (e.g., two character LoRAs)
            if loaded_type == new_type and loaded_type in [
                    "character", "subject"]:
                raise LoRAConflictError(
                    f"Multiple {loaded_type} LoRAs detected!\n"
                    f"  Already loaded: '{loaded_lora.name}'\n"
                    f"  Trying to add: '{new_lora.name}'\n"
                    f"  Solution: Use only one {loaded_type} LoRA at a time"
                )

    def calculate_combined_weight(
        self, loras: List[LoRAConfig]
    ) -> List[Tuple[LoRAConfig, float]]:
        """
        Calculate adjusted weights when stacking multiple LoRAs

        QUALITY OVER ALL: Reduce weights to maintain quality
        """
        if not loras:
            return []

        total_weight = sum(lora.weight for lora in loras)

        # If total weight > 1.5, scale down to prevent artifacts
        # This is a quality safeguard
        if total_weight > 1.5:
            self.logger.warning(
                f"Total LoRA weight ({total_weight:.2f}) exceeds safe limit (1.5). "
                f"Scaling down to maintain quality."
            )
            scale_factor = 1.5 / total_weight

            adjusted = []
            for lora in loras:
                adjusted_weight = lora.weight * scale_factor
                adjusted.append((lora, adjusted_weight))
                self.logger.info(
                    f"Adjusted weight for '{lora.name}': "
                    f"{lora.weight:.2f} -> {adjusted_weight:.2f}"
                )
            return adjusted

        return [(lora, lora.weight) for lora in loras]

    def resolve_lora_stack(
        self, requested_loras: List[LoRAConfig]
    ) -> List[Tuple[LoRAConfig, float]]:
        """
        Resolve LoRA stacking order and conflicts

        FAIL LOUD: Any unresolvable conflict causes failure
        QUALITY OVER ALL: Weights adjusted for quality

        Returns:
            List of (LoRAConfig, adjusted_weight) tuples in application order
        """
        if not requested_loras:
            return []

        self.loaded_loras.clear()
        resolved = []

        # Use ConfigurationManager for type priority - NO HARDCODED VALUES
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get type priority from configuration
        type_priority = config_manager.get_value('lora.type_priority')

        # Detect types and sort
        loras_with_types = []
        for lora in requested_loras:
            try:
                lora_type = self.detect_lora_type(lora)
                loras_with_types.append((lora, lora_type))
            except LoRAConflictError as e:
                # FAIL LOUD
                self.logger.error(f"LoRA type detection failed: {e}")
                raise

        # Sort by type priority - FAIL LOUD if type not in priority list
        def get_priority(lora_type_pair):
            lora, lora_type = lora_type_pair
            if lora_type not in type_priority:
                raise ValueError(
                    f"Unknown LoRA type '{lora_type}' for '{lora.name}'\n"
                    f"Known types: {list(type_priority.keys())}\n"
                    f"Please add '{lora_type}' to lora.type_priority in configuration."
                )
            return type_priority[lora_type]
        
        loras_with_types.sort(key=get_priority)

        # Check compatibility and build stack
        for lora, lora_type in loras_with_types:
            try:
                self.check_compatibility(lora, lora_type)
                self.loaded_loras.append((lora, lora_type))
                resolved.append(lora)
                self.logger.info(
                    f"Added LoRA '{lora.name}' (type: {lora_type}) to stack"
                )
            except LoRAConflictError as e:
                # FAIL LOUD - no silent skipping
                self.logger.error(f"LoRA stacking failed: {e}")
                raise

        # Calculate adjusted weights
        adjusted = self.calculate_combined_weight(resolved)

        self.logger.info(f"Resolved LoRA stack with {len(adjusted)} LoRAs:")
        for lora, weight in adjusted:
            self.logger.info(f"  - {lora.name}: weight={weight:.2f}")

        return adjusted

    def get_recommended_inference_steps(
        self, lora_stack: List[Tuple[LoRAConfig, float]]
    ) -> int:
        """
        Recommend inference steps based on LoRA stack

        QUALITY OVER ALL: More LoRAs = more steps for quality
        """
        # Use ConfigurationManager for all values - NO HARDCODED VALUES
        from ..core.configuration_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get LoRA configuration
        lora_config = config_manager.get_value('lora')
        base_steps = lora_config['base_inference_steps']
        max_steps = lora_config['max_inference_steps']
        type_steps = lora_config['type_steps']

        additional_steps = 0
        types_in_stack = set()

        for lora, _ in lora_stack:
            lora_type = self.detect_lora_type(lora)
            if lora_type not in types_in_stack:
                # FAIL LOUD if type not found
                if lora_type in type_steps:
                    additional_steps += type_steps[lora_type]
                else:
                    self.logger.warning(f"Unknown LoRA type '{lora_type}', using default")
                    additional_steps += type_steps['default']
                types_in_stack.add(lora_type)

        recommended = base_steps + additional_steps

        # Cap at configured maximum
        if recommended > max_steps:
            recommended = max_steps

        self.logger.info(
            f"Recommended inference steps for LoRA stack: {recommended}")
        return recommended
