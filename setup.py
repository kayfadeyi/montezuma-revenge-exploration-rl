from setuptools import setup, find_packages

setup(
    name="montezuma_rl",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "gymnasium[atari]>=1.0.0",
        "gymnasium[other]>=1.0.0",
        "torch>=1.9.0",
        "opencv-python>=4.5.3",
        "numpy>=2.2.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.3",
        "ale-py>=0.10.1",
        "moviepy>=2.1.1",
        "pillow>=9.2.0,<11.0",
        "objgraph>=3.6.2"
    ],
)

