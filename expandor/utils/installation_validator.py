"""
Installation validation utilities for Expandor
Checks dependencies, hardware, and provides fix instructions
"""

import errno
import importlib
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..utils.logging_utils import setup_logger


class InstallationValidator:
    """
    Validates Expandor installation and dependencies
    FAIL LOUD: Reports all issues clearly with solutions
    QUALITY OVER ALL: Ensures proper setup for best results
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize installation validator"""
        self.logger = logger or setup_logger(__name__)
        self.issues = []
        self.warnings = []
        self.info = {}

    def validate_installation(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Validate complete Expandor installation

        Args:
            verbose: Show detailed output

        Returns:
            Dictionary with validation results
        """
        self.logger.info("Validating Expandor installation...")

        # Clear previous results
        self.issues.clear()
        self.warnings.clear()
        self.info.clear()

        # Run all checks
        self._check_python_version()
        self._check_cuda_availability()
        self._check_required_packages()
        self._check_optional_packages()
        self._check_system_resources()
        self._check_expandor_components()
        self._check_permissions()

        # Compile results
        results = {
            "valid": len(self.issues) == 0,
            "issues": self.issues.copy(),
            "warnings": self.warnings.copy(),
            "info": self.info.copy(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "expandor_version": self._get_expandor_version(),
        }

        # Print summary if verbose
        if verbose:
            self._print_validation_summary(results)

        return results

    def _check_python_version(self):
        """Check Python version compatibility"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]

        if current_version < min_version:
            self.issues.append(
                {
                    "category": "Python Version",
                    "issue": f"Python {current_version[0]}.{current_version[1]} is too old",
                    "solution": f"Upgrade to Python {min_version[0]}.{min_version[1]} or newer",
                    "command": "pyenv install 3.8.0 && pyenv global 3.8.0  # or use your system's package manager",
                }
            )
        else:
            self.info["python_version"] = (
                f"{
                    current_version[0]}.{
                    current_version[1]}"
            )

    def _check_cuda_availability(self):
        """Check CUDA availability and version"""
        try:
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()

                self.info["cuda_version"] = cuda_version
                self.info["gpu_count"] = device_count

                # Get GPU details
                gpus = []
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(
                        i).total_memory / (1024**3)  # GB
                    gpus.append({"index": i, "name": gpu_name,
                                 "memory_gb": f"{gpu_memory:.1f}"})

                self.info["gpus"] = gpus

                # Check VRAM
                if device_count > 0:
                    total_vram = sum(
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(device_count)
                    )
                    total_vram_gb = total_vram / (1024**3)

                    if total_vram_gb < 8:
                        self.warnings.append(
                            {
                                "category": "GPU Memory",
                                "warning": f"Low VRAM detected: {
                                    total_vram_gb:.1f}GB",
                                "impact": "Limited to smaller resolutions or CPU offload mode",
                                "solution": "Use --strategy cpu_offload or --vram-limit flags",
                            })

                    self.info["total_vram_gb"] = f"{total_vram_gb:.1f}"
            else:
                # No CUDA - check why
                try:
                    import torch

                    if not hasattr(torch.cuda, "is_available"):
                        self.issues.append(
                            {
                                "category": "CUDA Support",
                                "issue": "PyTorch installed without CUDA support",
                                "solution": "Reinstall PyTorch with CUDA support",
                                "command": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                            })
                    else:
                        # CUDA not available on system
                        self.warnings.append(
                            {
                                "category": "GPU",
                                "warning": "No CUDA-capable GPU detected",
                                "impact": "Will use CPU mode (much slower)",
                                "solution": "Install NVIDIA GPU and CUDA drivers, or use cloud GPU services",
                            })
                except ImportError as e:
                    self.logger.debug(f"PyTorch import check: {e}")
                    # PyTorch check handled in check_dependencies()

        except Exception as e:
            self.issues.append(
                {
                    "category": "CUDA Check",
                    "issue": f"Failed to check CUDA: {e}",
                    "solution": "Ensure PyTorch is properly installed",
                    "command": "pip install --upgrade torch",
                }
            )

    def _check_required_packages(self):
        """Check required Python packages"""
        required_packages = {
            "torch": {
                "min_version": "2.0.0",
                "install": "pip install torch>=2.0.0",
                "critical": True,
            },
            "PIL": {
                "module": "PIL",
                "package": "pillow",
                "min_version": "9.0.0",
                "install": "pip install pillow>=9.0.0",
                "critical": True,
            },
            "numpy": {
                "min_version": "1.21.0",
                "install": "pip install numpy>=1.21.0",
                "critical": True,
            },
            "tqdm": {
                "min_version": "4.64.0",
                "install": "pip install tqdm>=4.64.0",
                "critical": True,
            },
            "yaml": {
                "module": "yaml",
                "package": "pyyaml",
                "min_version": "5.4.0",
                "install": "pip install pyyaml>=5.4.0",
                "critical": True,
            },
            "huggingface_hub": {
                "min_version": "0.16.0",
                "install": "pip install huggingface_hub>=0.16.0",
                "critical": False,
            },
        }

        missing_critical = []
        missing_optional = []

        for package_name, config in required_packages.items():
            module_name = config.get("module", package_name)
            package_install_name = config.get("package", package_name)

            try:
                module = importlib.import_module(module_name)

                # Check version if specified
                if "min_version" in config and hasattr(module, "__version__"):
                    current_version = module.__version__
                    min_version = config["min_version"]

                    if self._compare_versions(
                            current_version, min_version) < 0:
                        issue = {
                            "category": "Package Version",
                            "issue": f"{package_name} {current_version} is older than required {min_version}",
                            "solution": f"Upgrade {package_name}",
                            "command": config["install"],
                        }

                        if config.get("critical", True):
                            self.issues.append(issue)
                        else:
                            self.warnings.append(issue)

            except ImportError:
                issue = {
                    "category": "Missing Package",
                    "issue": f"{package_name} is not installed",
                    "solution": f"Install {package_name}",
                    "command": config["install"],
                }

                if config.get("critical", True):
                    self.issues.append(issue)
                    missing_critical.append(package_install_name)
                else:
                    self.warnings.append(issue)
                    missing_optional.append(package_install_name)

        # Provide combined install command for convenience
        if missing_critical:
            self.issues.append(
                {
                    "category": "Quick Fix",
                    "issue": "Install all missing critical packages",
                    "solution": "Run this command",
                    "command": f"pip install {' '.join(missing_critical)}",
                }
            )

    def _check_optional_packages(self):
        """Check optional packages for enhanced functionality"""
        optional_packages = {
            "diffusers": {
                "min_version": "0.25.0",
                "install": "pip install diffusers>=0.25.0",
                "feature": "AI model support",
            },
            "transformers": {
                "min_version": "4.35.0",
                "install": "pip install transformers>=4.35.0",
                "feature": "Text encoder support",
            },
            "accelerate": {
                "min_version": "0.25.0",
                "install": "pip install accelerate>=0.25.0",
                "feature": "Training and optimization",
            },
            "xformers": {
                "min_version": "0.0.20",
                "install": "pip install xformers",
                "feature": "Memory-efficient attention",
            },
            "cv2": {
                "module": "cv2",
                "package": "opencv-python",
                "install": "pip install opencv-python",
                "feature": "Advanced image processing",
            },
            "scipy": {
                "min_version": "1.9.0",
                "install": "pip install scipy>=1.9.0",
                "feature": "Frequency analysis for artifact detection",
            },
        }

        available_features = []

        for package_name, config in optional_packages.items():
            module_name = config.get("module", package_name)

            try:
                importlib.import_module(module_name)
                available_features.append(config["feature"])
            except ImportError:
                self.warnings.append(
                    {
                        "category": "Optional Package",
                        "warning": f"{package_name} not installed",
                        "impact": f"Missing feature: {config['feature']}",
                        "solution": config["install"],
                    }
                )

        if available_features:
            self.info["available_features"] = available_features

    def _check_system_resources(self):
        """Check system resources (disk space, memory)"""
        import psutil

        # Check disk space
        try:
            cache_dir = Path.home() / ".cache" / "huggingface"
            if cache_dir.exists():
                disk_usage = psutil.disk_usage(str(cache_dir))
                free_gb = disk_usage.free / (1024**3)

                if free_gb < 50:
                    self.warnings.append(
                        {
                            "category": "Disk Space",
                            "warning": f"Low disk space: {
                                free_gb:.1f}GB free",
                            "impact": "May not have space for model downloads",
                            "solution": "Free up disk space or change cache directory",
                        })

                self.info["disk_free_gb"] = f"{free_gb:.1f}"
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            # Non-critical - continue without disk space check

        # Check RAM
        try:
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)

            if total_ram_gb < 16:
                self.warnings.append(
                    {
                        "category": "System Memory",
                        "warning": f"Low RAM: {
                            total_ram_gb:.1f}GB",
                        "impact": "May have issues with large images or CPU mode",
                        "solution": "Close other applications or upgrade RAM",
                    })

            self.info["ram_gb"] = f"{total_ram_gb:.1f}"
        except Exception as e:
            self.logger.warning(f"Could not check system memory: {e}")
            # Non-critical - continue without memory check

    def _check_expandor_components(self):
        """Check Expandor-specific components"""
        try:
            # Check if strategies are available
            from ..strategies import STRATEGY_REGISTRY

            strategy_count = len(STRATEGY_REGISTRY)
            self.info["available_strategies"] = list(STRATEGY_REGISTRY.keys())

            if strategy_count < 4:
                self.issues.append(
                    {
                        "category": "Expandor Components",
                        "issue": f"Only {strategy_count} strategies available",
                        "solution": "Reinstall Expandor",
                        "command": "pip install --force-reinstall expandor",
                    }
                )

            # Check configuration
            from ..config import UserConfigManager

            config_path = UserConfigManager().config_path

            if not config_path.parent.exists():
                self.warnings.append(
                    {
                        "category": "Configuration",
                        "warning": "Config directory doesn't exist",
                        "impact": "Will use default settings",
                        "solution": "Run 'expandor --setup' to create configuration",
                    })

        except ImportError as e:
            self.issues.append(
                {
                    "category": "Expandor Installation",
                    "issue": f"Core components missing: {e}",
                    "solution": "Reinstall Expandor",
                    "command": "pip install --force-reinstall expandor",
                }
            )

    def _check_permissions(self):
        """Check file system permissions"""
        test_paths = [
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".config" / "expandor",
            Path.cwd() / "expandor_output",
        ]

        for path in test_paths:
            try:
                # Try to create directory
                path.mkdir(parents=True, exist_ok=True)

                # Try to write a test file
                test_file = path / ".expandor_test"
                test_file.write_text("test")
                test_file.unlink()

            except PermissionError:
                self.issues.append(
                    {
                        "category": "Permissions",
                        "issue": f"No write permission for: {path}",
                        "solution": f"Grant write permissions",
                        "command": f"chmod -R u+w {path}",
                    }
                )
            except OSError as e:
                if e.errno != errno.EEXIST:
                    self.logger.debug(f"Directory check for {path}: {e}")
                # Directory might not need to exist yet

    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare version strings (returns -1, 0, 1)"""
        try:
            from packaging import version

            v1 = version.parse(version1)
            v2 = version.parse(version2)

            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
        except BaseException:
            # Fallback to simple comparison
            return - \
                1 if version1 < version2 else (1 if version1 > version2 else 0)

    def _get_expandor_version(self) -> str:
        """Get Expandor version"""
        try:
            import expandor

            return getattr(expandor, "__version__", "unknown")
        except BaseException:
            return "not installed"

    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print formatted validation summary"""
        print("\n" + "=" * 60)
        print("EXPANDOR INSTALLATION VALIDATION REPORT")
        print("=" * 60)

        # System Info
        print("\n📊 SYSTEM INFORMATION:")
        print(f"  • Platform: {results['platform']}")
        print(
            f"  • Python: {
                results.get(
                    'info',
                    {}).get(
                    'python_version',
                    'unknown')}"
        )
        print(f"  • Expandor: {results['expandor_version']}")

        # GPU Info
        if results["cuda_available"]:
            print(f"\n🎮 GPU INFORMATION:")
            info = results.get("info", {})
            print(f"  • CUDA Version: {info.get('cuda_version', 'unknown')}")
            print(f"  • Total VRAM: {info.get('total_vram_gb', 'unknown')}GB")

            for gpu in info.get("gpus", []):
                print(
                    f"  • GPU {
                        gpu['index']}: {
                        gpu['name']} ({
                        gpu['memory_gb']}GB)"
                )
        else:
            print(f"\n⚠️  No CUDA GPU detected - will use CPU mode")

        # Issues
        if results["issues"]:
            print(f"\n❌ CRITICAL ISSUES ({len(results['issues'])}):")
            for i, issue in enumerate(results["issues"], 1):
                print(f"\n  {i}. {issue['category']}:{issue['issue']}")
                print(f"     Fix: {issue['solution']}")
                if "command" in issue:
                    print(f"     Run: {issue['command']}")

        # Warnings
        if results["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"\n  • {warning['category']}:{warning['warning']}")
                print(f"    Impact: {warning['impact']}")
                print(f"    Suggestion: {warning['solution']}")

        # Available Features
        features = results.get("info", {}).get("available_features", [])
        if features:
            print(f"\n✅ AVAILABLE FEATURES:")
            for feature in features:
                print(f"  • {feature}")

        # Summary
        print("\n" + "=" * 60)
        if results["valid"]:
            print("✅ INSTALLATION IS VALID - Ready to use Expandor!")
        else:
            print("❌ INSTALLATION HAS ISSUES - Please fix critical issues above")
        print("=" * 60 + "\n")
