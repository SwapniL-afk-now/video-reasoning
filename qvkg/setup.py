from setuptools import setup, find_packages

setup(
    name="qvkg",
    version="0.1.0",
    description="Query-Conditioned Video Knowledge Graph",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.21.0",
        "torch>=2.0",
        "transformers>=4.40",
        "opencv-python-headless",
        "networkx",
        "numpy",
        "pydantic>=2.0",
        "tqdm",
        "h5py",
        "scipy",
        "scikit-learn",
        "soundfile",
        "av",
        "sortedcontainers",
        "einops",
        "pyyaml",
    ],
    extras_require={
        "whisper": ["faster-whisper"],
        "faiss-gpu": ["faiss-gpu-cu12"],
        "dev": ["pytest"],
    },
)
