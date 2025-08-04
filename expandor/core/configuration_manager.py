"""Singleton configuration manager - the ONLY way to access config"""

import copy
import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import jsonschema
from jsonschema import validate, ValidationError
try:
    # Try new referencing library (jsonschema >= 4.18)
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT7
    HAS_REFERENCING = True
except ImportError:
    # Fall back to deprecated RefResolver for older versions
    from jsonschema import RefResolver
    HAS_REFERENCING = False
from ..utils.config_loader import ConfigLoader
from ..utils.path_resolver import PathResolver
from ..core.exceptions import ExpandorError


class ConfigurationManager:
    """Singleton configuration manager - the ONLY way to access config"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if ConfigurationManager._initialized:
            return
        ConfigurationManager._initialized = True
        
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}
        self._user_overrides = {}
        self._env_overrides = {}
        self._runtime_overrides = {}
        self._config_loader = None
        self._master_config = {}
        self._schemas = {}
        self._schema_resolver = None
        self._path_resolver = PathResolver(self.logger)
        
        # Compatibility attributes for tests
        self._config = {}  # Alias for final merged config
        self.configs = {}  # Alias for backward compatibility
        
        # Load schemas first
        self._load_schemas()
        
        # Initialize on first use
        self._load_all_configurations()
    
    def _load_schemas(self):
        """Load all JSON schemas for validation - FAIL LOUD on errors"""
        self._schemas = {}
        schema_dir = Path(__file__).parent.parent / "config" / "schemas"
        
        # FAIL LOUD if schema directory doesn't exist
        if not schema_dir.exists():
            raise ValueError(
                f"Schema directory not found: {schema_dir}\n"
                f"This is a critical configuration error.\n"
                f"The schema directory must exist for proper validation."
            )
        
        # Find all schema files
        schema_files = list(schema_dir.glob("*.schema.json"))
        if not schema_files:
            # FAIL LOUD - schemas are required for validation
            raise ValueError(
                f"No schema files found in {schema_dir}\n"
                f"This is a critical configuration error.\n"
                f"Schema files are required for proper validation.\n"
                f"Please create at least master_defaults.schema.json\n"
                f"to define the configuration structure."
            )
        
        # Load each schema - FAIL LOUD on any error
        for schema_file in schema_files:
            try:
                with open(schema_file, 'r') as f:
                    schema_name = schema_file.stem.replace('.schema', '')
                    self._schemas[schema_name] = json.load(f)
            except json.JSONDecodeError as e:
                # FAIL LOUD on invalid JSON
                raise ValueError(
                    f"Invalid JSON in schema file {schema_file}:\n"
                    f"Error: {e}\n"
                    f"Please fix the schema file and try again."
                ) from e
            except Exception as e:
                # FAIL LOUD on any other error
                raise ValueError(
                    f"Failed to load schema {schema_file}:\n"
                    f"Error: {e}\n"
                    f"This is a critical error that prevents proper validation."
                ) from e
        
        # Create resolver for $ref resolution
        if self._schemas:
            if HAS_REFERENCING:
                # Use new referencing library
                resources = []
                # Add resources with both file:// URI and simple filename for resolution
                for schema_name, schema_content in self._schemas.items():
                    # Add with full file:// URI
                    full_uri = f"file://{schema_dir}/{schema_name}.schema.json"
                    resources.append((full_uri, Resource.from_contents(schema_content)))
                    # Also add with just filename for relative references
                    filename = f"{schema_name}.schema.json"
                    resources.append((filename, Resource.from_contents(schema_content)))
                    # And without .schema suffix for simpler references
                    simple_name = f"{schema_name}.json"
                    resources.append((simple_name, Resource.from_contents(schema_content)))
                
                self._schema_resolver = Registry().with_resources(resources)
            else:
                # Fall back to deprecated RefResolver
                base_uri = "file://" + str(schema_dir) + "/"
                # FAIL LOUD if base_schema is missing
                if 'base_schema' not in self._schemas:
                    raise ValueError(
                        "FATAL: base_schema not found in schemas!"
                        "This is a critical configuration error."
                    )
                self._schema_resolver = RefResolver(base_uri, self._schemas['base_schema'])
    
    def _validate_config(self, config_data: dict, schema_name: str):
        """Validate config against schema - FAIL LOUD on invalid"""
        if schema_name not in self._schemas:
            # FAIL LOUD - missing schema is a critical error
            raise ValueError(
                f"FATAL: No schema found for {schema_name}!\n"
                f"This is a critical configuration error.\n"
                f"Available schemas: {list(self._schemas.keys())}\n"
                f"Please ensure the schema file exists in the schemas directory."
            )
        
        try:
            if HAS_REFERENCING:
                # Use new referencing approach - create validator with registry
                from jsonschema import Draft7Validator
                validator_cls = Draft7Validator
                validator = validator_cls(
                    schema=self._schemas[schema_name],
                    registry=self._schema_resolver
                )
                validator.validate(config_data)
            else:
                # Use deprecated resolver approach
                validate(
                    instance=config_data,
                    schema=self._schemas[schema_name],
                    resolver=self._schema_resolver
                )
        except ValidationError as e:
            # FAIL LOUD with helpful error
            raise ValueError(
                f"Configuration validation failed for {schema_name}!\n"
                f"Error: {e.message}\n"
                f"Failed at path: {' -> '.join(str(p) for p in e.path)}\n"
                f"Schema rule: {e.schema}\n"
                f"Invalid value: {e.instance}\n\n"
                f"Please fix your configuration file and ensure all required fields are present."
            )
    
    def _load_all_configurations(self):
        """Load all configurations in proper hierarchy"""
        # 1. Load system defaults from package config dir
        config_dir = Path(__file__).parent.parent / "config"
        self._config_loader = ConfigLoader(config_dir, self.logger)
        
        # Load master defaults - the ONLY source of truth
        try:
            self._master_config = self._config_loader.load_config_file("master_defaults.yaml")
            self.logger.info("Loaded master configuration file")
            
            # Check version and migrate if needed
            self._check_and_migrate_config(self._master_config, "master")
            
            # Always validate master config - FAIL LOUD if schema missing
            # Skip validation only if no schemas loaded at all
            if self._schemas:
                if 'master_defaults' not in self._schemas:
                    # FAIL LOUD - master schema is required if any schemas exist
                    raise ValueError(
                        f"Master defaults schema not found!\n"
                        f"Expected: master_defaults.schema.json\n"
                        f"Available schemas: {list(self._schemas.keys())}\n"
                        f"The master configuration must have a schema for validation."
                    )
                # FAIL LOUD - schema validation is mandatory
                self._validate_config(self._master_config, 'master_defaults')
        except FileNotFoundError as e:
            # FAIL LOUD - master config is required
            raise ValueError(
                f"Master configuration file not found: master_defaults.yaml\n"
                f"This is a critical error. The master configuration must exist.\n"
                f"Expected location: {config_dir / 'master_defaults.yaml'}"
            ) from e
        
        # 2. Load user configuration
        self._load_user_config()
        
        # 3. Load environment overrides
        self._load_env_overrides()
        
        # 4. Build final config cache
        self._build_config_cache()
    
    
    def _load_user_config(self):
        """Load user configuration from standard locations"""
        # Build search paths, handling empty env var properly
        search_paths = []
        
        # Only add EXPANDOR_CONFIG_PATH if it's set and not empty
        env_path = os.environ.get("EXPANDOR_CONFIG_PATH", "")
        if env_path:
            search_paths.append(Path(env_path))
        
        # Add standard locations
        search_paths.extend([
            Path.home() / ".config" / "expandor" / "config.yaml",
            Path.cwd() / "expandor.yaml",
            Path("/etc/expandor/config.yaml")
        ])
        
        for path in search_paths:
            # Skip if path is a directory or doesn't exist
            if path.is_file():
                try:
                    with open(path, 'r') as f:
                        user_config = yaml.safe_load(f)
                        if user_config:
                            # Convert numeric strings to proper types
                            from ..utils.config_loader import ConfigLoader
                            loader = ConfigLoader(Path(), self.logger)
                            user_config = loader._convert_numeric_strings(user_config, path="")
                            # Check version and migrate if needed
                            self._check_and_migrate_config(user_config, "user", path)
                            self._user_overrides = user_config
                            self.logger.info(f"Loaded user config from {path}")
                            break
                except Exception as e:
                    self.logger.error(f"Failed to load user config from {path}: {e}")
            elif path.exists() and path.is_dir():
                # Skip directories silently
                self.logger.debug(f"Skipping directory {path} in config search")
    
    def _find_user_config(self) -> Optional[Path]:
        """Find user configuration file in standard locations"""
        # Build search paths, handling empty env var properly
        search_paths = []
        
        # Only add EXPANDOR_CONFIG_PATH if it's set and not empty
        env_path = os.environ.get("EXPANDOR_CONFIG_PATH", "")
        if env_path:
            search_paths.append(Path(env_path))
        
        # Add standard locations
        search_paths.extend([
            Path.home() / ".config" / "expandor" / "config.yaml",
            Path.cwd() / "expandor.yaml",
            Path("/etc/expandor/config.yaml")
        ])
        
        for path in search_paths:
            if path.is_file():
                return path
        return None
    
    def _load_env_overrides(self):
        """Load environment variable overrides (EXPANDOR_*)"""
        for key, value in os.environ.items():
            if key.startswith("EXPANDOR_"):
                # Convert EXPANDOR_PROCESSING_BATCH_SIZE to processing.batch_size
                # We need to be smarter about underscores - only convert section separators
                parts = key[9:].lower().split("_")
                
                # Try different combinations to find a valid key
                # Start with assuming first part is section, rest is key
                for i in range(1, len(parts)):
                    section = ".".join(parts[:i])
                    key_part = "_".join(parts[i:])
                    config_path = f"{section}.{key_part}"
                    
                    # Check if this path exists in master config
                    try:
                        self._get_nested_value(self._master_config, config_path)
                        # Found valid path!
                        break
                    except KeyError:
                        # Try next combination
                        continue
                else:
                    # Fallback: convert all underscores to dots
                    config_path = key[9:].lower().replace("_", ".")
                
                try:
                    # Try to parse as number/bool
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    elif "." in value:
                        parsed_value = float(value)
                    elif value.isdigit():
                        parsed_value = int(value)
                    else:
                        parsed_value = value
                    
                    self._set_nested_value(self._env_overrides, config_path, parsed_value)
                except Exception:
                    # Keep as string if parsing fails
                    self._set_nested_value(self._env_overrides, config_path, value)
    
    def has_key(self, key: str) -> bool:
        """Check if a configuration key exists"""
        try:
            self.get_value(key)
            return True
        except (ValueError, KeyError):
            return False
    
    def get_value(self, key: str, context: Optional[Dict] = None) -> Any:
        """
        Get config value - FAILS LOUD if not found
        
        Args:
            key: Dot-separated config key (e.g., 'strategies.progressive_outpaint.base_strength')
            context: Optional context for dynamic resolution
            
        Returns:
            Configuration value
            
        Raises:
            ValueError: If key not found (FAIL LOUD)
        """
        # Check runtime overrides first
        if context and 'override' in context:
            if key in context['override']:
                return context['override'][key]
        
        # Check config cache
        try:
            value = self._get_nested_value(self._config_cache, key)
            if value is not None:
                return value
        except KeyError:
            pass
        
        # FAIL LOUD - no silent defaults
        raise ValueError(
            f"Configuration key '{key}' not found!\n"
            f"This is a required configuration value with no default.\n"
            f"Solutions:\n"
            f"1. Add '{key}' to your config files\n" 
            f"2. Set environment variable EXPANDOR_{key.upper().replace('.', '_')}\n"
            f"3. Check config file syntax for errors"
        )
    
    def _merge_config(self, base: dict, override: dict):
        """Deep merge override into base config"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _get_nested_value(self, config: dict, key: str) -> Any:
        """Get value from nested dict using dot notation"""
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Key '{k}' not found in path '{key}'")
        return value
    
    def _set_nested_value(self, config: dict, key: str, value: Any):
        """Set value in nested dict using dot notation"""
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def _build_config_cache(self):
        """Build final config cache from all sources"""
        # Start with master config
        self._config_cache = copy.deepcopy(self._master_config)
        
        # Apply user overrides
        self._merge_config(self._config_cache, self._user_overrides)
        
        # Apply environment overrides (highest priority)
        self._merge_config(self._config_cache, self._env_overrides)
        
        # Update compatibility attributes
        self._config = self._config_cache
        self.configs = {'master': self._config_cache}
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get complete config for a strategy"""
        base_key = f"strategies.{strategy_name}"
        try:
            return self._get_nested_value(self._config_cache, base_key)
        except KeyError:
            raise ValueError(
                f"No configuration found for strategy '{strategy_name}'\n"
                f"Available strategies: {list(self._config_cache['strategies'].keys() if 'strategies' in self._config_cache else [])}"
            )
    
    def get_processor_config(self, processor_name: str) -> Dict[str, Any]:
        """Get complete config for a processor"""
        base_key = f"processors.{processor_name}"
        try:
            return self._get_nested_value(self._config_cache, base_key)
        except KeyError:
            raise ValueError(
                f"No configuration found for processor '{processor_name}'\n"
                f"Available processors: {list(self._config_cache['processors'].keys() if 'processors' in self._config_cache else [])}"
            )
    
    def get_path(self, path_key: str, create: bool = True, 
                 path_type: str = "directory") -> Path:
        """
        Get resolved path from configuration
        
        Args:
            path_key: Configuration key for path (e.g., 'paths.cache_dir')
            create: Create if doesn't exist
            path_type: 'directory' or 'file'
            
        Returns:
            Resolved Path object
        """
        # Get defaults if not provided
        if create is None:
            create = self.get_value('constants.cli.default_create')
        if path_type is None:
            path_type = self.get_value('constants.cli.default_path_type')
            
        path_config = self.get_value(path_key)
        return self._path_resolver.resolve_path(path_config, create, path_type)
    
    def _check_and_migrate_config(self, config: Dict[str, Any], config_type: str, 
                                  path: Optional[Path] = None):
        """
        Check configuration version and migrate if needed
        
        Args:
            config: Configuration dictionary
            config_type: Type of config ("master" or "user")
            path: Path to config file (for user configs)
        """
        # Current expected version
        CURRENT_VERSION = "2.0"
        
        # Import ConfigMigrator once
        from ..utils.config_migrator import ConfigMigrator
        config_dir = Path(__file__).parent.parent / "config"
        migrator = ConfigMigrator(config_dir)
        
        # Get config version - use migrator to detect if not present
        if 'version' in config:
            config_version = str(config['version'])
        else:
            # Use ConfigMigrator to intelligently detect version
            config_version = migrator.detect_version(config) or '1.0'
        
        if config_version == CURRENT_VERSION:
            return  # No migration needed
        
        # FAIL LOUD on version mismatch for master config
        if config_type == "master" and config_version != CURRENT_VERSION:
            raise ValueError(
                f"Master configuration version mismatch!\n"
                f"Expected version: {CURRENT_VERSION}\n"
                f"Found version: {config_version}\n"
                f"This indicates the configuration system is out of sync.\n"
                f"Please update master_defaults.yaml to version {CURRENT_VERSION}"
            )
        
        # For user configs, attempt migration
        if config_type == "user":
            self.logger.warning(
                f"User configuration at {path} has version {config_version}, "
                f"expected {CURRENT_VERSION}. Attempting migration..."
            )
            
            # Perform migration
            try:
                # Note: ConfigMigrator.migrate() method handles its own file I/O,
                # but we need to call the migration method that accepts config as parameter
                # For now, we'll need to adapt the migration approach
                migration_key = (config_version, CURRENT_VERSION)
                if migration_key in migrator.migration_map:
                    migration_func = migrator.migration_map[migration_key]
                    migrated_config = migration_func(config)
                else:
                    raise ValueError(
                        f"No migration path from version {config_version} to {CURRENT_VERSION}. "
                        f"Available paths: {list(migrator.migration_map.keys())}"
                    )
                
                # Update the config in-place
                config.clear()
                config.update(migrated_config)
                
                # Save migrated config back to file
                if path:
                    import shutil
                    from datetime import datetime
                    
                    # Backup original
                    backup_path = path.with_suffix(
                        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    shutil.copy2(path, backup_path)
                    self.logger.info(f"Backed up original config to {backup_path}")
                    
                    # Save migrated config
                    with open(path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    self.logger.info(f"Saved migrated config to {path}")
                
                self.logger.info(
                    f"Successfully migrated configuration from version "
                    f"{config_version} to {CURRENT_VERSION}"
                )
                
            except ImportError:
                # Migration script not available - FAIL LOUD
                raise ValueError(
                    f"Configuration migration required but migration script not found!\n"
                    f"User config at {path} has version {config_version}, "
                    f"but system expects {CURRENT_VERSION}.\n"
                    f"Please run: expandor --migrate-config\n"
                    f"Or manually update your configuration to version {CURRENT_VERSION}"
                )
            except Exception as e:
                # Migration failed - FAIL LOUD
                raise ValueError(
                    f"Configuration migration failed!\n"
                    f"Config at {path} has version {config_version}, "
                    f"but migration to {CURRENT_VERSION} failed.\n"
                    f"Error: {e}\n"
                    f"Please check your configuration or recreate it with: "
                    f"expandor --init-config"
                )