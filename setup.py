"""
Setup script for NPT Model package.
Package configuration and installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="npt-model",
    version="0.1.0",
    author="NPT Team",
    description="Neural Pre-Trained Language Model - A complete LLM training framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/npt-model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "distributed": [
            "deepspeed>=0.12.0",
            "torch-distributed-rpc",
        ],
        "inference": [
            "onnxruntime>=1.17.0",
            "tensorflow>=2.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "npt-train=training.pretraining.train:main",
            "npt-inference=inference.server:main",
        ],
    },
)
