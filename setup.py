from setuptools import setup, find_packages

setup(
    name="montezuma_rl",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "gymnasium[atari]>=0.29.1",
        "torch>=1.9.0",
        "opencv-python>=4.5.3",
        "numpy>=1.19.5",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.3"
    ],
)

