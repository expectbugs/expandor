"""Path resolution and validation for Expandor"""

import os
from pathlib import Path
from typing import Optional, Union
import logging


class PathResolver:
    """Resolves and validates all file paths"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}
    
    def resolve_path(self, path_config: Union[str, Path], 
                     create: bool = True,
                     path_type: str = "directory") -> Path:
        """
        Resolve path with smart expansion
        
        Args:
            path_config: Path configuration string
            create: Create directory if it doesn't exist
            path_type: "directory" or "file"
            
        Returns:
            Resolved Path object
            
        Raises:
            ValueError: If path invalid or can't be created
        """
        if not path_config:
            raise ValueError("Path configuration cannot be None or empty")
        
        # Convert to string for processing
        path_str = str(path_config)
        
        # Check cache
        cache_key = f"{path_str}:{create}:{path_type}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Expand variables
        path_str = os.path.expandvars(path_str)
        path_str = os.path.expanduser(path_str)
        
        # Handle relative paths
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        
        # Resolve to absolute
        try:
            path = path.resolve()
        except Exception as e:
            raise ValueError(
                f"Failed to resolve path '{path_config}'\n"
                f"Error: {e}"
            )
        
        # Create if requested
        if create:
            try:
                if path_type == "directory":
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    # For files, create parent directory
                    path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ValueError(
                    f"Permission denied creating path '{path}'\n"
                    f"Check directory permissions or use a different path"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create path '{path}'\n"
                    f"Error: {e}"
                )
        
        # Validate exists if not creating
        elif not path.exists() and path_type == "directory":
            raise ValueError(
                f"Path does not exist: '{path}'\n"
                f"Original config: '{path_config}'\n"
                f"Set create=True to create it automatically"
            )
        
        # Cache and return
        self._cache[cache_key] = path
        self.logger.debug(f"Resolved path '{path_config}' to '{path}'")
        return path
    
    def get_writable_dir(self, preferred_paths: list, 
                         purpose: str = "data") -> Path:
        """
        Find first writable directory from list
        
        Args:
            preferred_paths: List of paths in preference order
            purpose: Description of what this is for (for error messages)
            
        Returns:
            First writable directory
            
        Raises:
            ValueError: If no writable directory found
        """
        for path_config in preferred_paths:
            try:
                path = self.resolve_path(path_config, create=True)
                # Test write permission
                test_file = path / ".expandor_write_test"
                test_file.touch()
                test_file.unlink()
                return path
            except Exception as e:
                self.logger.debug(f"Path '{path_config}' not writable: {e}")
                continue
        
        raise ValueError(
            f"No writable directory found for {purpose}\n"
            f"Tried paths: {preferred_paths}\n"
            f"Please ensure at least one location is writable"
        )