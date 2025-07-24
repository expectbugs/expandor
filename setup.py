"""
Expandor - Universal Image Resolution Adaptation System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="expandor",
    version="0.5.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Universal image resolution and aspect ratio adaptation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/expandor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "huggingface_hub>=0.16.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "diffusers": [
            "diffusers>=0.25.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
        ],
        "all": [
            "diffusers>=0.25.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "xformers>=0.0.20",
            "scipy>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "expandor=expandor.cli.main:main",
        ],
    },
)
