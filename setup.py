"""Setup script for comfy-commander package."""

from setuptools import setup

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="comfy-commander",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for programmatically running ComfyUI workloads either locally or remotely",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/comfy-commander",
    package_dir={"": "src"},
    packages=["comfy_commander"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio",
            "pytest-sugar",
            "syrupy",
        ],
    },
)

