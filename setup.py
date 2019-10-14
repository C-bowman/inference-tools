from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="inference-tools",
    version="0.5.0",
    author="Chris Bowman",
    author_email="chris.bowman.physics@gmail.com",
    description="A collection of python tools for Bayesian data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/C-bowman/inference-tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)