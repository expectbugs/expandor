#!/usr/bin/env python3
"""
Validate all configuration files in Expandor
Ensures configuration consistency and completeness
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import yaml
import jsonschema
from jsonschema import validate, ValidationError
from collections import defaultdict


class ConfigValidator:
    """Comprehensive configuration validator"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
        # Required configuration files
        self.required_files = {
            'master_defaults.yaml': 'Master configuration with all defaults',
            'quality_presets.yaml': 'Quality preset definitions',
            'strategies.yaml': 'Strategy configurations',
            'controlnet_config.yaml': 'ControlNet configuration',
        }
        
        # Required top-level sections in master_defaults.yaml
        self.required_sections = {
            'version': 'Configuration version for migration',
            'quality_presets': 'Quality preset definitions',
            'strategies': 'Strategy-specific configurations',
            'processing': 'Processing parameters',
            'output': 'Output settings',
            'paths': 'Path configurations',
            'vram': 'VRAM management settings',
            'quality_thresholds': 'Quality validation thresholds',
            'system': 'System settings',
        }
        
        # Value ranges for validation
        self.value_ranges = {
            'denoising_strength': (0.0, 1.0),
            'guidance_scale': (0.0, 30.0),
            'num_inference_steps': (1, 200),
            'overlap_ratio': (0.0, 1.0),
            'compression_level': (0, 9),
            'jpeg_quality': (1, 100),
            'batch_size': (1, 32),
            'tile_size': (64, 2048),
            'window_size': (32, 1024),
            'vram_limit_mb': (0, 131072),  # Up to 128GB
        }
        
        # Type requirements
        self.type_requirements = {
            'denoising_strength': float,
            'guidance_scale': float,
            'num_inference_steps': int,
            'overlap_ratio': float,
            'tile_size': int,
            'batch_size': int,
            'enable_artifacts_check': bool,
            'save_stages': bool,
            'verbose': bool,
        }

    def validate(self) -> bool:
        """Run all validation checks"""
        print("Validating Expandor configuration files...")
        print("=" * 80)
        
        # Check required files exist
        self.check_required_files()
        
        # Load and validate master_defaults.yaml
        master_config = self.validate_master_config()
        
        # Validate individual config files
        self.validate_quality_presets()
        self.validate_strategies_config()
        self.validate_controlnet_config()
        
        # Cross-file validation
        if master_config:
            self.validate_cross_references(master_config)
        
        # Check for missing configuration values
        self.check_completeness(master_config)
        
        # Validate JSON schemas if present
        self.validate_schemas()
        
        # Report results
        return self.report_results()
    
    def check_required_files(self):
        """Check that all required configuration files exist"""
        print("\n## Checking required files...")
        
        for filename, description in self.required_files.items():
            file_path = self.config_dir / filename
            if not file_path.exists():
                self.add_error(f"Missing required file: {filename} - {description}")
            else:
                self.stats['files_found'] += 1
                print(f"✓ Found {filename}")
    
    def validate_master_config(self) -> Dict[str, Any]:
        """Validate master_defaults.yaml structure and content"""
        print("\n## Validating master_defaults.yaml...")
        
        config_path = self.config_dir / 'master_defaults.yaml'
        if not config_path.exists():
            self.add_error("master_defaults.yaml not found - cannot continue validation")
            return {}
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Check version
            if 'version' not in config:
                self.add_error("Missing 'version' field in master_defaults.yaml")
            else:
                print(f"✓ Configuration version: {config['version']}")
            
            # Check required sections
            for section, description in self.required_sections.items():
                if section not in config:
                    self.add_error(f"Missing required section '{section}': {description}")
                else:
                    self.stats['sections_found'] += 1
            
            # Validate nested structure
            self.validate_config_structure(config, 'master_defaults.yaml')
            
            return config
            
        except yaml.YAMLError as e:
            self.add_error(f"YAML parsing error in master_defaults.yaml: {e}")
            return {}
        except Exception as e:
            self.add_error(f"Error loading master_defaults.yaml: {e}")
            return {}
    
    def validate_config_structure(self, config: Dict[str, Any], filename: str):
        """Validate configuration structure and values"""
        def check_value(key: str, value: Any, path: str):
            # Check type requirements
            if key in self.type_requirements:
                expected_type = self.type_requirements[key]
                if not isinstance(value, expected_type):
                    self.add_error(
                        f"Type mismatch in {filename} at {path}: "
                        f"'{key}' should be {expected_type.__name__}, got {type(value).__name__}"
                    )
            
            # Check value ranges
            if key in self.value_ranges and isinstance(value, (int, float)):
                min_val, max_val = self.value_ranges[key]
                if not min_val <= value <= max_val:
                    self.add_error(
                        f"Value out of range in {filename} at {path}: "
                        f"'{key}' = {value} (should be {min_val} <= value <= {max_val})"
                    )
            
            # Check for None values (violates FAIL LOUD)
            if value is None:
                self.add_warning(
                    f"None value found in {filename} at {path}: "
                    f"'{key}' - should have explicit default"
                )
        
        def traverse(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    check_value(key, value, new_path)
                    traverse(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    traverse(item, f"{path}[{i}]")
        
        traverse(config)
    
    def validate_quality_presets(self):
        """Validate quality_presets.yaml"""
        print("\n## Validating quality_presets.yaml...")
        
        config_path = self.config_dir / 'quality_presets.yaml'
        if not config_path.exists():
            self.add_warning("quality_presets.yaml not found")
            return
        
        try:
            with open(config_path) as f:
                presets = yaml.safe_load(f)
            
            required_presets = ['ultra', 'high', 'balanced', 'fast']
            for preset_name in required_presets:
                if preset_name not in presets:
                    self.add_error(f"Missing required preset '{preset_name}' in quality_presets.yaml")
                else:
                    # Validate preset structure
                    preset = presets[preset_name]
                    self.validate_preset_structure(preset, preset_name)
            
            print(f"✓ Found {len(presets)} quality presets")
            
        except Exception as e:
            self.add_error(f"Error validating quality_presets.yaml: {e}")
    
    def validate_preset_structure(self, preset: Dict[str, Any], preset_name: str):
        """Validate individual preset structure"""
        required_fields = [
            'generation.denoising_strength',
            'generation.guidance_scale', 
            'generation.num_inference_steps',
            'quality.enable_artifacts_check',
            'processing.save_stages',
        ]
        
        for field_path in required_fields:
            parts = field_path.split('.')
            current = preset
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    self.add_error(
                        f"Missing required field '{field_path}' in preset '{preset_name}'"
                    )
                    break
                current = current[part]
    
    def validate_strategies_config(self):
        """Validate strategies.yaml"""
        print("\n## Validating strategies.yaml...")
        
        config_path = self.config_dir / 'strategies.yaml'
        if not config_path.exists():
            self.add_warning("strategies.yaml not found")
            return
        
        try:
            with open(config_path) as f:
                strategies = yaml.safe_load(f)
            
            # Check each strategy has required fields
            required_strategies = [
                'direct_upscale', 'progressive_outpaint', 
                'tiled_expansion', 'swpo', 'cpu_offload'
            ]
            
            for strategy_name in required_strategies:
                if strategy_name not in strategies:
                    self.add_error(f"Missing strategy '{strategy_name}' in strategies.yaml")
            
            print(f"✓ Found {len(strategies)} strategies")
            
        except Exception as e:
            self.add_error(f"Error validating strategies.yaml: {e}")
    
    def validate_controlnet_config(self):
        """Validate controlnet_config.yaml"""
        print("\n## Validating controlnet_config.yaml...")
        
        config_path = self.config_dir / 'controlnet_config.yaml'
        if not config_path.exists():
            self.add_warning("controlnet_config.yaml not found")
            return
        
        try:
            with open(config_path) as f:
                controlnet = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['defaults', 'extractors', 'models', 'pipelines']
            for section in required_sections:
                if section not in controlnet:
                    self.add_error(f"Missing section '{section}' in controlnet_config.yaml")
            
            print("✓ ControlNet configuration validated")
            
        except Exception as e:
            self.add_error(f"Error validating controlnet_config.yaml: {e}")
    
    def validate_cross_references(self, master_config: Dict[str, Any]):
        """Validate references between configuration files"""
        print("\n## Validating cross-references...")
        
        # Check that quality presets referenced in master match quality_presets.yaml
        if 'quality_global' in master_config and 'default_preset' in master_config['quality_global']:
            default_preset = master_config['quality_global']['default_preset']
            presets_file = self.config_dir / 'quality_presets.yaml'
            
            if presets_file.exists():
                with open(presets_file) as f:
                    presets = yaml.safe_load(f)
                    if default_preset not in presets:
                        self.add_error(
                            f"Default preset '{default_preset}' referenced in master_defaults.yaml "
                            f"not found in quality_presets.yaml"
                        )
    
    def check_completeness(self, master_config: Dict[str, Any]):
        """Check for configuration completeness"""
        print("\n## Checking configuration completeness...")
        
        # List of all configuration keys that should exist
        required_configs = [
            # Paths
            'paths.cache_dir',
            'paths.output_dir', 
            'paths.temp_dir',
            
            # Processing
            'processing.save_stages',
            'processing.verbose',
            'processing.batch_size',
            
            # Quality
            'quality_global.default_preset',
            'quality_thresholds.seam_detection_threshold',
            'quality_thresholds.texture_corruption_threshold',
            
            # VRAM
            'vram.estimation.latent_multiplier',
            'vram.offloading.enable_sequential',
            
            # Output
            'output.formats.png.compression',
            'output.formats.jpeg.quality',
        ]
        
        missing_configs = []
        for config_path in required_configs:
            parts = config_path.split('.')
            current = master_config
            
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    missing_configs.append(config_path)
                    break
                current = current[part]
        
        if missing_configs:
            for config in missing_configs:
                self.add_error(f"Missing required configuration: {config}")
        else:
            print("✓ All required configuration keys present")
    
    def validate_schemas(self):
        """Validate JSON schemas if present"""
        print("\n## Checking JSON schemas...")
        
        schema_dir = self.config_dir / 'schemas'
        if not schema_dir.exists():
            self.add_warning("No schemas directory found - skipping schema validation")
            return
        
        # Find all schema files
        schema_files = list(schema_dir.glob('*.json'))
        if not schema_files:
            self.add_warning("No JSON schema files found")
            return
        
        for schema_file in schema_files:
            try:
                with open(schema_file) as f:
                    schema = json.load(f)
                
                # Validate the schema itself
                jsonschema.Draft7Validator.check_schema(schema)
                print(f"✓ Valid schema: {schema_file.name}")
                
            except Exception as e:
                self.add_error(f"Invalid schema {schema_file.name}: {e}")
    
    def add_error(self, message: str):
        """Add an error message"""
        self.errors.append(message)
        self.stats['errors'] += 1
        print(f"❌ ERROR: {message}")
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)
        self.stats['warnings'] += 1
        print(f"⚠️  WARNING: {message}")
    
    def report_results(self) -> bool:
        """Generate final report"""
        print("\n" + "=" * 80)
        print("## Validation Summary\n")
        
        print(f"Files checked: {self.stats['files_found']}/{len(self.required_files)}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Warnings: {self.stats['warnings']}")
        
        if self.errors:
            print("\n### Errors that must be fixed:")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")
        
        if self.warnings:
            print("\n### Warnings to consider:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
        
        # Save detailed report
        report_path = Path(__file__).parent / 'config_validation_report.md'
        with open(report_path, 'w') as f:
            f.write("# Configuration Validation Report\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Files checked: {self.stats['files_found']}/{len(self.required_files)}\n")
            f.write(f"- Errors: {self.stats['errors']}\n")
            f.write(f"- Warnings: {self.stats['warnings']}\n\n")
            
            if self.errors:
                f.write("## Errors\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            if self.warnings:
                f.write("## Warnings\n")
                for warning in self.warnings:
                    f.write(f"- {warning}\n")
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Return success if no errors
        return len(self.errors) == 0


def main():
    """Main entry point"""
    # Determine config directory
    script_path = Path(__file__).resolve()
    config_dir = script_path.parent.parent / 'expandor' / 'config'
    
    if not config_dir.exists():
        print(f"Error: Config directory not found at {config_dir}")
        sys.exit(1)
    
    # Create validator and run
    validator = ConfigValidator(config_dir)
    success = validator.validate()
    
    # Exit with appropriate code
    if success:
        print("\n✅ Configuration validation passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Configuration validation failed with {validator.stats['errors']} errors!")
        sys.exit(1)


if __name__ == '__main__':
    main()