"""
Integration tests for Expandor CLI
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


class TestCLICommands:
    """Test CLI commands end-to-end"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_image(self, temp_dir):
        """Create test image"""
        img = Image.new("RGB", (1024, 768), color=(100, 150, 200))
        img_path = temp_dir / "test_image.png"
        img.save(img_path)
        return img_path

    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test config file"""
        config = {
            "models": {
                "sdxl": {
                    "model_id": "test/mock-sdxl",
                    "dtype": "float16",
                    "device": "cuda",
                }
            },
            "default_quality": "balanced",
        }
        config_path = temp_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def run_cli(self, args):
        """Run CLI command and capture output"""
        cmd = [sys.executable, "-m", "expandor.cli.main"] + args
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
        )
        return result

    def test_help_command(self):
        """Test --help flag"""
        result = self.run_cli(["--help"])
        assert result.returncode == 0
        assert "Universal Image Resolution Adaptation System" in result.stdout
        assert "--resolution" in result.stdout
        assert "--quality" in result.stdout

    def test_version_command(self):
        """Test --version flag"""
        result = self.run_cli(["--version"])
        assert result.returncode == 0
        assert "0.4.0" in result.stdout

    def test_single_image_expansion(self, test_image, temp_dir):
        """Test basic single image expansion"""
        output_path = temp_dir / "output.png"
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2048x1536",
                "--output",
                str(output_path),
                "--dry-run",  # Dry run to avoid needing real pipeline
            ]
        )

        assert result.returncode == 0
        assert "Would process" in result.stdout
        assert "1024x768" in result.stdout  # Original size
        assert "2048x1536" in result.stdout  # Target size

    def test_resolution_preset(self, test_image, temp_dir):
        """Test resolution preset parsing"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "4K",
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "3840x2160" in result.stdout

    def test_resolution_multiplier(self, test_image, temp_dir):
        """Test resolution multiplier"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2x",
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "2048x1536" in result.stdout  # 2x of 1024x768

    def test_quality_presets(self, test_image, temp_dir):
        """Test different quality presets"""
        for quality in ["fast", "balanced", "high", "ultra"]:
            result = self.run_cli(
                [
                    str(test_image),
                    "--resolution",
                    "2K",
                    "--quality",
                    quality,
                    "--output",
                    str(temp_dir / f"output_{quality}.png"),
                    "--dry-run",
                ]
            )

            assert result.returncode == 0
            assert f"quality preset: {quality}" in result.stdout.lower()

    def test_batch_processing(self, temp_dir):
        """Test batch processing with wildcards"""
        # Create multiple test images
        for i in range(3):
            img = Image.new("RGB", (800, 600), color=(i * 50, i * 50, i * 50))
            img.save(temp_dir / f"test_{i}.png")

        output_dir = temp_dir / "output"
        result = self.run_cli(
            [
                str(temp_dir / "test_*.png"),
                "--resolution",
                "1600x1200",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "Would process 3 images" in result.stdout

    def test_custom_config(self, test_image, test_config, temp_dir):
        """Test custom config file"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--config",
                str(test_config),
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "Using config:" in result.stdout

    def test_lora_stacking(self, test_image, temp_dir):
        """Test LoRA stacking arguments"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--lora",
                "style_anime",
                "--lora",
                "detail_enhance",
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "LoRAs:" in result.stdout
        assert "style_anime" in result.stdout
        assert "detail_enhance" in result.stdout

    def test_invalid_resolution(self, test_image, temp_dir):
        """Test invalid resolution handling"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "invalid_res",
                "--output",
                str(temp_dir / "output.png"),
            ]
        )

        assert result.returncode != 0
        assert (
            "Invalid resolution format" in result.stderr
            or "Unknown resolution" in result.stderr
        )

    def test_missing_input(self, temp_dir):
        """Test missing input file error"""
        result = self.run_cli(
            [
                str(temp_dir / "nonexistent.png"),
                "--resolution",
                "2K",
                "--output",
                str(temp_dir / "output.png"),
            ]
        )

        assert result.returncode != 0
        assert "not found" in result.stderr.lower()

    def test_strategy_selection(self, test_image, temp_dir):
        """Test explicit strategy selection"""
        strategies = ["direct", "progressive", "swpo", "tiled"]

        for strategy in strategies:
            result = self.run_cli(
                [
                    str(test_image),
                    "--resolution",
                    "4K",
                    "--strategy",
                    strategy,
                    "--output",
                    str(temp_dir / f"output_{strategy}.png"),
                    "--dry-run",
                ]
            )

            assert result.returncode == 0
            assert f"strategy: {strategy}" in result.stdout.lower()

    def test_artifact_detection_toggle(self, test_image, temp_dir):
        """Test artifact detection toggle"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--no-artifact-detection",
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "artifact detection: disabled" in result.stdout.lower()

    def test_save_stages(self, test_image, temp_dir):
        """Test save stages option"""
        stage_dir = temp_dir / "stages"
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--save-stages",
                "--stage-dir",
                str(stage_dir),
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "stages directory:" in result.stdout.lower()
        assert str(stage_dir) in result.stdout

    def test_verbose_output(self, test_image, temp_dir):
        """Test verbose output"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--output",
                str(temp_dir / "output.png"),
                "--verbose",
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        # Verbose output should have more details
        assert len(result.stdout) > 500  # Arbitrary threshold

    def test_prompt_and_negative_prompt(self, test_image, temp_dir):
        """Test prompt arguments"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--prompt",
                "beautiful landscape, high quality",
                "--negative-prompt",
                "blurry, low quality",
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "prompt:" in result.stdout.lower()
        assert "beautiful landscape" in result.stdout
        assert "negative prompt:" in result.stdout.lower()
        assert "blurry" in result.stdout

    def test_seed_reproducibility(self, test_image, temp_dir):
        """Test seed argument"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "2K",
                "--seed",
                "12345",
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "seed: 12345" in result.stdout.lower()

    def test_vram_limit(self, test_image, temp_dir):
        """Test VRAM limit argument"""
        result = self.run_cli(
            [
                str(test_image),
                "--resolution",
                "8K",
                "--vram-limit",
                "4000",  # 4GB
                "--output",
                str(temp_dir / "output.png"),
                "--dry-run",
            ]
        )

        assert result.returncode == 0
        assert "vram limit: 4000" in result.stdout.lower()

    def test_output_formats(self, test_image, temp_dir):
        """Test different output formats"""
        formats = ["png", "jpg", "webp"]

        for fmt in formats:
            result = self.run_cli(
                [
                    str(test_image),
                    "--resolution",
                    "2K",
                    "--output-format",
                    fmt,
                    "--output",
                    str(temp_dir / f"output.{fmt}"),
                    "--dry-run",
                ]
            )

            assert result.returncode == 0
            assert f"format: {fmt}" in result.stdout.lower()


class TestCLIErrorHandling:
    """Test CLI error handling and recovery"""

    def run_cli(self, args):
        """Run CLI command and capture output"""
        cmd = [sys.executable, "-m", "expandor.cli.main"] + args
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent
        )
        return result

    def test_conflicting_output_options(self, temp_dir):
        """Test error when both --output and --output-dir specified"""
        result = self.run_cli(
            [
                "test.png",
                "--resolution",
                "2K",
                "--output",
                str(temp_dir / "output.png"),
                "--output-dir",
                str(temp_dir),
            ]
        )

        assert result.returncode != 0
        assert "Cannot specify both" in result.stderr

    def test_invalid_quality_preset(self, temp_dir):
        """Test invalid quality preset"""
        result = self.run_cli(
            ["test.png", "--resolution", "2K", "--quality", "invalid_quality"]
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_invalid_model(self, temp_dir):
        """Test invalid model selection"""
        result = self.run_cli(
            ["test.png", "--resolution", "2K", "--model", "invalid_model"]
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()
